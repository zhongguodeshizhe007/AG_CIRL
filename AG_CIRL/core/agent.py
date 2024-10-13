import multiprocessing
import math
import time
import os
import torch
import numpy as np
from utils.replay_memory import Memory
from utils.torch import to_device

os.environ["OMP_NUM_THREADS"] = "1"

'''该函数主要用于在工作进程中采样样本，并将结果存储到样本存储器memory中。在采样过程中，它还会记录一些统计信息，并将它们保存到日志log中。如果指定了队列queue，则将工作进程的结果放入queue中，以便主进程可以获取并合并所有工作进程的结果。如果未指定queue，则直接返回结果'''
def collect_samples(pid, queue, env, policy, custom_reward, mean_action, render, running_state, min_batch_size):
    if pid > 0: 
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        if hasattr(env, 'np_random'):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
            env.env.np_random.seed(env.env.np_random.randint(5000) * pid)
    '''初始化一些变量和日志信息，包括样本存储器memory、步骤数num_steps、总奖励total_reward、最小奖励min_reward、最大奖励max_reward和回合数num_episodes'''
    log = dict()
    memory = Memory() # 创建一个样本存储器memory
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size: # min_batch_size=8192; 在未达到要求的最小min_batch_size之前，循环执行以下操作：
        state, des = env.reset() 
        reward_episode = 0 # 初始化回合奖励 reward_episode
        des_var = torch.tensor(des).long().unsqueeze(0) # 将目的地D 转换为张量长整数类型，并扩展维度，来自于专家轨迹的目的地终点
        for t in range(50): # 在每个时间步内进行如下操作：
            state_var = torch.tensor(state).long().unsqueeze(0)
            with torch.no_grad():
                if mean_action: 
                    action = torch.argmax(policy.get_action_prob(state_var, des_var)).unsqueeze(0).numpy() 
                else: # 在本文mean_action=False
                    action = policy.select_action(state_var, des_var)[0].numpy() 
            action = int(action) 
            next_state, reward, done = env.step(action) 
            reward_episode += reward 
            mask = 0 if (done or t == 49) else 1
            bad_mask = 0 if (next_state == env.pad_idx or t == 49) else 1
            memory.push(state, des, action, next_state, reward, mask, bad_mask) # 根据当前状态、des、动作、下一个状态、奖励和结束标志，将样本存储到 memory 中
            if done: # 如果达到了回合结束或时间步长达到上限，则终止当前回合
                break
            state = next_state # 更新当前状态为下一个状态
        # 更新步骤数 num_steps，回合数 num_episodes，总奖励 total_reward，最小奖励 min_reward 和最大奖励 max_reward
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)
    # 将步骤数、回合数、总奖励、平均奖励、最大奖励和最小奖励保存到日志 log 中
    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    # 如果队列 queue 不为 None，则将工作进程的标识符、样本存储器 memory 和日志 log 放入队列中
    if queue is not None:
        queue.put([pid, memory, log])
    else: # 如果队列 queue 为 None，则返回样本存储器 memory 和日志 log
        return memory, log # memory中存储了(state, des, action, next_state, reward, mask, bad_mask)

'''用于收集使用给定策略在环境中生成的路线（trajectories）'''
def collect_routes_with_OD(pid, batch_od, queue, env, policy, custom_reward,
                    mean_action, render, running_state): # mean_action=True
    if pid > 0: #如果pid大于0，则根据pid设置随机种子，以确保在不同进程中生成的随机数不同。这样做是为了避免多个并行进程生成相同的随机序列
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        if hasattr(env, 'np_random'):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
            env.env.np_random.seed(env.env.np_random.randint(5000) * pid)

    trajs = [] # 创建一个空列表trajs，用于存储生成的路线
    '''
    对于每个样本（每一行），重置环境，并将起始状态和目标位置传递给环境的reset函数。然后，创建一个空列表traj，用于存储该样本的路线，并将起始状态state作为字符串添加到路线中
    '''
    for i in range(batch_od.shape[0]):
        state, des = env.reset(int(batch_od[i, 0]), int(batch_od[i, 1])) # 将当前batch_od[i, 0]设置为state，将batch_od[i, 1]设置为des
        reward_episode = 0
        des_var = torch.tensor(des).long().unsqueeze(0)
        traj = [str(state)]
        '''
        在每个时间步内，根据当前状态state选择动作。如果mean_action为True，则根据策略的动作概率选择具有最高概率的动作。否则，根据策略选择动作。
        然后，使用选择的动作与环境进行交互，获得下一个状态next_state。如果下一个状态为-1，表示达到了终止状态，跳出循环。
        将下一个状态添加到路线traj中，并检查是否已完成一个完整的路线（根据done的值）。如果是，则跳出循环。
        '''
        for t in range(50):
            state_var = torch.tensor(state).long().unsqueeze(0)
            with torch.no_grad():
                if mean_action: # 根据策略的动作概率选择具有最高概率的动作
                    action = torch.argmax(policy.get_action_prob(state_var, des_var)).unsqueeze(0).numpy()
                else:
                    action = policy.select_action(state_var, des_var)[0].numpy()
            action = int(action)
            next_state, _, done = env.step(action) #
            if next_state == -1:
                break
            traj.append(str(next_state))
            if done:
                break
            state = next_state
        trajs.append(traj) # 将生成的路线traj添加到列表trajs中
    if queue is not None: # 如果提供了一个队列queue，则将进程ID和路线列表作为一个元组放入队列中。否则，直接返回路线列表。
        queue.put([pid, trajs])
    else:
        return trajs


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:
    def __init__(self, env, policy, device, custom_reward=None, running_state=None, num_threads=1):
        self.env = env
        self.policy = policy # policy_net
        self.device = device # device(type='cuda')
        self.custom_reward = custom_reward # None
        self.running_state = running_state # None
        self.num_threads = num_threads # 4, 表示线程数

    def collect_samples(self, min_batch_size, mean_action=False, render=False): # min_batch_size=8192; mean_action=False
        t_start = time.time() # 获取当前时间 t_start
        to_device(torch.device('cpu'), self.policy) # 将self.policy移动到CPU设备，以确保样本收集在 CPU 上进行；；to_device函数：return [x.to(device) for x in args]
        self.policy.to_device(torch.device('cpu')) # 将 self.policy 移动到 CPU 设备，以确保样本收集在 CPU 上进行
        # 确定每个线程需要处理的最小批次大小thread_batch_size，通过将总的最小批次大小min_batch_size除以线程数self.num_threads并向下取整得到
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads)) # 确定每个线程需要处理的最小批次大小
        queue = multiprocessing.Queue() # 创建一个用于进程间通信和数据传递的队列 queue，用于从工作进程中收集样本
        workers = [] # 创建一个工作进程列表workers，用于保存每个工作进程的引用
        '''总的来说，通过使用多线程和进程间通信的机制，将样本收集任务分配给多个工作进程并行处理，从而提高了样本收集的效率'''
        for i in range(self.num_threads - 1): # 使用一个循环来创建并启动多个工作进程（线程）
        # 构造工作进程的参数 worker_args，包括工作进程的标识符 i+1、队列queue、环境self.env、策略模型self.policy、自定义奖励函数self.custom_reward、
        # 平均动作标志mean_action、渲染标志False、运行状态self.running_state和线程批次大小thread_batch_size
            worker_args = (i + 1, queue, self.env, self.policy, self.custom_reward, mean_action,
                           False, self.running_state, thread_batch_size) # 构造工作进程的参数 worker_args
            # 使用 multiprocessing.Process 创建工作进程，并将 collect_samples 方法作为目标函数，传递参数 worker_args
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args)) # 允许创建并行运行的进程，并提供了一种将函数作为参数传递给新进程的方法
        for worker in workers:
            worker.start() # 使用 start() 方法启动每个工作进程，使它们开始执行任务
        # 在主进程中，调用 collect_samples 方法进行样本收集，将工作进程的标识符设为0，将队列queue设为None，并传递其他参数
        # memory中存储了(state, des, action, next_state, reward, mask, bad_mask)
        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, mean_action,
                                      render, self.running_state, thread_batch_size) # # mean_action=False
        # 创建工作进程日志和样本存储器的空列表 worker_logs 和 worker_memories，长度为工作进程数
        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers: # 从队列中获取每个工作进程的结果，并将它们分别存储在 worker_memories 和 worker_logs 列表中
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories: # 将每个工作进程的样本存储器 worker_memory 追加到主进程的样本存储器 memory 中
            memory.append(worker_memory)
        batch = memory.sample() 
        if self.num_threads > 1: # 如果线程数大于1，将主进程的日志 log 与所有工作进程的日志合并为一个日志列表 log_list
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy) # 将设备重新设置为目标设备 self.device，以确保后续的计算在正确的设备上进行
        self.policy.to_device(self.device)
        t_end = time.time() # 获取当前时间 t_end
        '''返回合并后的样本和一些日志信息，包括样本采集时间、动作的平均值、最小值和最大值等'''
        '''创建日志信息 log，包括样本采集时间、动作的平均值、最小值和最大值等'''
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log # 返回从memory抽取的样本batch 和日志 log

    def collect_routes_with_OD(self, target_od, mean_action=False, render=False): # mean_action=True；test_od为：df[['ori', 'des']].values
        to_device(torch.device('cpu'), self.policy)
        self.policy.to_device(torch.device('cpu'))
        thread_batch_size = int(math.ceil(target_od.shape[0] / self.num_threads)) # test_od为：df[['ori', 'des']].values
        
        batch_od = [target_od[i*thread_batch_size:min((i+1)*thread_batch_size, target_od.shape[0])]for i in range(self.num_threads)] # 成为几个分块列表了
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads - 1):
            worker_args = (i + 1, batch_od[i+1], queue, self.env, self.policy, self.custom_reward, mean_action,
                           False, self.running_state) # mean_action=True
            workers.append(multiprocessing.Process(target=collect_routes_with_OD, args=worker_args))
        for worker in workers:
            worker.start()

        trajs = collect_routes_with_OD(0, batch_od[0], None, self.env, self.policy, self.custom_reward, mean_action,
                                      render, self.running_state) # mean_action=True；batch_od[0]表示batch_od列表中的第一个元素

        worker_trajs = [None] * len(workers)
        for _ in workers:
            pid, worker_traj = queue.get()
            worker_trajs[pid - 1] = worker_traj
        for worker_traj in worker_trajs:
            trajs = trajs + worker_traj
        to_device(self.device, self.policy)
        self.policy.to_device(self.device)
        return trajs