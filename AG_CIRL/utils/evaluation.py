import editdistance
from nltk.translate.bleu_score import sentence_bleu
from scipy.spatial import distance
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch

smoothie = SmoothingFunction().method1
device = torch.device("cpu")

'''该函数通过遍历测试轨迹列表来创建起始-目标位置对字典。对于每个轨迹，它获取轨迹的第一个位置和最后一个位置作为起始-目标位置对的键。
然后，它检查该键是否已经存在于字典中，如果存在，则将当前轨迹索引添加到对应的值列表中；如果不存在，则创建新的键值对，键为起始-目标位置对，值为包含当前轨迹索引的列表。
最后，函数返回创建的起始-目标位置对字典'''
def create_od_set(test_trajs):
    test_od_dict = {}
    for i in range(len(test_trajs)):
        if (test_trajs[i][0], test_trajs[i][-1]) in test_od_dict.keys():
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])].append(i)
        else:
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])] = [i]
    return test_od_dict

'''
该函数用于计算测试轨迹和学习者轨迹之间的编辑距离。它接受测试轨迹列表、学习者轨迹列表和测试起始-目标位置对字典作为输入。它首先根据起始-目标位置对从测试轨迹字典中获取对应的轨迹列表，
并将其转换为字符串形式。然后，对于每个学习者轨迹，它计算与所有测试轨迹的编辑距离，并选择最小的编辑距离作为该学习者轨迹的最佳编辑距离。最后，它返回所有最佳编辑距离的平均值。
'''
def evaluate_edit_dist(test_trajs, learner_trajs, test_od_dict): # 用于计算测试轨迹和学习者轨迹之间的编辑距离。
    edit_dist_list = [] # 该函数首先创建一个空列表 edit_dist_list，用于存储每对轨迹之间的最小编辑距离
    for od in test_od_dict.keys(): # 然后，对于测试轨迹字典中的每个起始-目标位置对 od，它获取对应的测试轨迹索引列表 idx_list
        idx_list = test_od_dict[od] # 将测试轨迹根据起始-目标位置对进行组织。它将每个轨迹转换为一个字符串，其中位置之间使用下划线 _ 连接，然后将这些字符串放入一个集合中，以去除重复的轨迹。最后，它将每个字符串再次拆分为位置列表，以便后续的编辑距离计算
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs: # 获取对应的学习者轨迹列表 learner_od_trajs，该列表包含了与测试轨迹相对应的学习者轨迹。
            min_edit_dist = 1.0
            '''接下来，它遍历学习者轨迹列表，并与每个测试轨迹进行比较，计算它们之间的编辑距离。对于每个学习者轨迹 learner，它会遍历测试轨迹集合中的每个测试轨迹 test，
                使用编辑距离函数 editdistance.eval() 计算它们之间的编辑距离，并将最小的编辑距离更新为 min_edit_dist'''
            for test in test_od_trajs:
                edit_dist = editdistance.eval(test, learner) / len(test)
                min_edit_dist = edit_dist if edit_dist < min_edit_dist else min_edit_dist
            edit_dist_list.append(min_edit_dist)
    return np.mean(edit_dist_list)

'''
该函数用于评估BLEU分数指标。它接受测试轨迹列表、学习者轨迹列表和测试起始-目标位置对字典作为输入。它首先根据起始-目标位置对从测试轨迹字典中获取对应的轨迹列表，
并将其转换为字符串形式。然后，对于每个学习者轨迹，它计算与所有测试轨迹的BLEU分数，并将分数添加到列表中。最后，它返回所有BLEU分数的平均值。
'''
def evaluate_bleu_score(test_trajs, learner_trajs, test_od_dict):
    bleu_score_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        # get unique reference
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            # print(test_od_trajs)
            # print(learner)
            bleu_score = sentence_bleu(test_od_trajs, learner, smoothing_function=smoothie)
            bleu_score_list.append(bleu_score)
    return np.mean(bleu_score_list)

'''
该函数用于评估数据集之间的分布差异（Jensen-Shannon距离）。它接受测试轨迹列表和学习者轨迹列表作为输入。首先，它将测试轨迹列表转换为字符串形式，
并计算测试轨迹字符串的概率分布。然后，它将学习者轨迹列表转换为字符串形式，并根据测试轨迹字符串的概率分布计算学习者轨迹字符串的概率分布。
最后，它计算测试轨迹字符串概率分布和学习者轨迹字符串概率分布之间的Jensen-Shannon距离，并返回该距离。
'''
def evaluate_dataset_dist(test_trajs, learner_trajs):
    test_trajs_str = ['_'.join(traj) for traj in test_trajs]
    # print('test trajs str len', len(test_trajs_str))
    test_trajs_set = set(test_trajs_str)
    # print('test trajs set len', len(test_trajs_set))
    test_trajs_dict = dict(zip(list(test_trajs_set), range(len(test_trajs_set))))
    test_trajs_label = [test_trajs_dict[traj] for traj in test_trajs_str]
    test_trajs_label.append(0)
    test_p = np.histogram(test_trajs_label)[0] / len(test_trajs_label)

    pad_idx = len(test_trajs_set)
    learner_trajs_str = ['_'.join(traj) for traj in learner_trajs]
    learner_trajs_label = [test_trajs_dict.get(traj, pad_idx) for traj in learner_trajs_str]
    learner_p = np.histogram(learner_trajs_label)[0] / len(learner_trajs_label)
    return distance.jensenshannon(test_p, learner_p)

'''
该函数用于评估轨迹的对数概率。它接受一个测试轨迹列表和模型作为输入。对于每个测试轨迹，它计算轨迹中每个状态的动作的对数概率，
并将它们相加得到总的对数概率。最后，它返回所有轨迹的对数概率的平均值。
'''
def evaluate_log_prob(test_traj, model):
    log_prob_list = []
    for episode in test_traj:
        des = torch.LongTensor([episode[-1].next_state]).long().to(device)
        log_prob = 0
        for x in episode:
            with torch.no_grad():
                next_prob = torch.log(model.get_action_prob(torch.LongTensor([x.cur_state]).to(device), des)).squeeze()
            next_prob_np = next_prob.detach().cpu().numpy()
            log_prob += next_prob_np[x.action]
        log_prob_list.append(log_prob)
    # print(np.mean(log_prob_list))
    return np.mean(log_prob_list)

'''
该函数用于评估在训练数据上的编辑距离。它接受训练轨迹列表和学习者轨迹列表作为输入，并根据训练轨迹创建起始-目标位置对字典。
然后，它使用evaluate_edit_dist()函数计算在训练数据上的编辑距离，并返回该值。
'''
def evaluate_train_edit_dist(train_traj, learner_traj):
    ''' 这个函数用于保留在训练数据上具有最佳编辑距离性能的训练epoch '''
    test_od_dict = create_od_set(train_traj)
    edit_dist = evaluate_edit_dist(train_traj, learner_traj, test_od_dict)
    return edit_dist

'''
 该函数用于评估多个指标，包括编辑距离、BLEU分数和数据集分布差异。它接受测试轨迹列表和学习者轨迹列表作为输入，并根据测试轨迹创建起始-目标位置对字典。
 然后，它使用evaluate_edit_dist()函数计算编辑距离，使用evaluate_bleu_score()函数计算BLEU分数，使用evaluate_dataset_dist()函数计算数据集分布差异，并将它们打印出来。
 最后，它返回编辑距离、BLEU分数和数据集分布差异的值。
'''
def evaluate_metrics(test_traj, learner_traj):
    test_od_dict = create_od_set(test_traj)
    edit_dist = evaluate_edit_dist(test_traj, learner_traj, test_od_dict)
    bleu_score = evaluate_bleu_score(test_traj, learner_traj, test_od_dict)
    js_dist = evaluate_dataset_dist(test_traj, learner_traj)
    return edit_dist, bleu_score, js_dist

'''
该函数用于评估模型的性能。它接受目标起始-目标位置对、目标轨迹、模型、环境和链接数量作为输入。它首先计算第一个起始-目标位置对的转移概率矩阵，并使用贪心策略生成学习者轨迹。
然后，对于剩余的起始-目标位置对，它根据当前位置和目标位置计算转移概率矩阵，并使用贪心策略生成学习者轨迹。
最后，它使用evaluate_metrics()函数评估学习者轨迹和目标轨迹之间的编辑距离、BLEU分数和数据集分布差异，并将它们打印出来。
'''
def evaluate_model(target_od, target_traj, model, env, n_link=714):
    state_ts = torch.from_numpy(np.arange(n_link)).long().to(device)
    target_o, target_d = target_od[:, 0].tolist(), target_od[:, 1].tolist()
    learner_traj = []
    curr_ori, curr_des = target_o[0], target_d[0]
    des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
    action_prob = model.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # 714, 8
    state_action = env.state_action[:-1]
    action_prob[state_action == env.pad_idx] = 0.0
    transit_prob = np.zeros((n_link, n_link))
    from_st, ac = np.where(state_action != env.pad_idx)
    to_st = state_action[state_action != env.pad_idx]
    transit_prob[from_st, to_st] = action_prob[from_st, ac]
    sample_path = [str(curr_ori)]
    curr_state = curr_ori
    for _ in range(50):
        if curr_state == curr_des: break
        next_state = np.argmax(transit_prob[curr_state])
        sample_path.append(str(next_state))
        curr_state = next_state
    learner_traj.append(sample_path)
    for ori, des in zip(target_o[1:], target_d[1:]):
        if des == curr_des:
            if ori == curr_ori:
                learner_traj.append(sample_path)
                continue
            else:
                curr_ori = ori
        else:
            curr_ori, curr_des = ori, des
            des_ts = (torch.ones_like(state_ts) * curr_des).to(device)
            action_prob = model.get_action_prob(state_ts, des_ts).detach().cpu().numpy()  # 714, 8
            state_action = env.state_action[:-1]
            action_prob[state_action == env.pad_idx] = 0.0
            transit_prob = np.zeros((n_link, n_link))
            from_st, ac = np.where(state_action != env.pad_idx)
            to_st = state_action[state_action != env.pad_idx]
            transit_prob[from_st, to_st] = action_prob[from_st, ac]
        sample_path = [str(curr_ori)]
        curr_state = curr_ori
        for _ in range(50):
            if curr_state == curr_des: break
            next_state = np.argmax(transit_prob[curr_state])
            sample_path.append(str(next_state))
            curr_state = next_state
        learner_traj.append(sample_path)
        import pandas as pd
        cv = 0 
        size = 100
        # 创建一个DataFrame
        data = {'path': learner_traj}
        df = pd.DataFrame(data)
        csv_filename = "../four_metrics/bc_CV%d_size%d_learner_traj.csv" % (cv, size)
        # 将DataFrame保存为CSV文件
        df.to_csv(csv_filename, index=False)
    edit_dist, bleu_score, js_dist = evaluate_metrics(target_traj, learner_traj)
    return edit_dist, bleu_score, js_dist
