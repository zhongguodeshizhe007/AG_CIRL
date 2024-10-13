from collections import namedtuple # 创建一个具名元组类型，并为该类型指定字段的名称
import torch
import numpy as np
import pandas as pd

Step = namedtuple('Step', ['cur_state', 'action', 'next_state', 'reward', 'done']) # 创建一个具名元组类型，并为该类型指定字段的名称，eg. Person = namedtuple('Person', ['name', 'age', 'gender'])


class RoadWorld(object):
    """
    Environment
    """

    def __init__(self, network_path, edge_path, pre_reset=None, origins=None,
                 destinations=None, k=8):
        self.network_path = network_path
        self.netin = origins
        self.netout = destinations
        self.k = k
        self.max_route_length = 0

        netconfig = np.load(self.network_path) 
        netconfig = pd.DataFrame(netconfig, columns=["from", "con", "to"]) # 'from'和'to'表示link ID；'con'表示方向
        netconfig_dict = {}
        for i in range(len(netconfig)):
            fromid, con, toid = netconfig.loc[i] # con表示方向：0,1,2,3,4,5,6,7

            if fromid in netconfig_dict.keys():
                netconfig_dict[fromid][con] = toid
            else:
                netconfig_dict[fromid] = {}
                netconfig_dict[fromid][con] = toid
        self.netconfig = netconfig_dict 
        # define states and actions
        edge_df = pd.read_csv(edge_path, header=0, usecols=['n_id']) 

        # self.terminal = len(self.states)  # add a terminal state for destination
        self.states = edge_df['n_id'].tolist() # 
        self.pad_idx = len(self.states) 
        self.states.append(self.pad_idx) 
        self.actions = range(k)  # k 动作的个数，也即是8个方向

        self.n_states = len(self.states) # 7 所有可能的state的数量
        self.n_actions = len(self.actions) # 8个方向；所有可能的action的数量
        print('state的数量：', self.n_states) # 
        print('action的数量：', self.n_actions) # 8

        self.rewards = [0 for _ in range(self.n_states)] # 创建一个列表，并给奖励赋初始值为0
        
        self.state_action_pair = sum([[(s, a) for a in self.get_action_list(s)] for s in self.states], []) # 获取当前所有state-action对的合理组合。注意：有些是不合理的，即：当前link下可能没有左拐之类的，这类型的状态-动作对组合不被纳入
        self.num_sapair = len(self.state_action_pair) # 状态-动作对的个数
        print('state-action pair的数量：', self.num_sapair)

        self.sapair_idxs = self.state_action_pair  # 表示状态-动作对的索引列表
        # 下面一行代码表示策略的掩码，即记录策略中哪些状态-动作对是可行的或被允许的；掩码值为 0 表示对应的状态-动作对不可行或不被允许，非零值表示对应的状态-动作对是可行的或被允许的。
        # 可以在实施策略时考虑到限制条件或约束，只选择允许的状态-动作对，从而满足环境或任务的要求
        self.policy_mask = np.zeros([self.n_states, self.n_actions], dtype=np.int32) 

        self.state_action = np.ones([self.n_states, self.n_actions], dtype=np.int32) * self.pad_idx 
        print('policy mask', self.policy_mask.shape)
        '''下面循环的解释：
        将 self.policy_mask 中对应的元素设为 1，表示该状态-动作对是合法的（可能会被选中），只有具有合理性的状态-动作对才会被设为 1，其他位置保持为默认值 0。
        同时，代码还将 self.state_action 中对应的元素设为 self.netconfig[s][a]，即使用 self.netconfig 中对应的值更新该状态-动作对的索引'''
        for s, a in self.sapair_idxs: 
            self.policy_mask[s, a] = 1 
            self.state_action[s, a] = self.netconfig[s][a] 
            
        self.cur_state = None
        self.cur_des = None

        if pre_reset is not None: # 在文中执行以下两行代码
            self.od_list = pre_reset[0] 
            self.od_dist = pre_reset[1] 
   
    def reset(self, st=None, des=None):
        if st is not None and des is not None:
            self.cur_state, self.cur_des = st, des
        else: 
            od_idx = np.random.choice(self.od_list, 1, p=self.od_dist) 
            ori, des = od_idx[0].split('_') 
            self.cur_state, self.cur_des = int(ori), int(des)
        return self.cur_state, self.cur_des 

    def get_reward(self, state):
        return self.rewards[state]
    '''该函数接受一个action作为输入，并返回下一个状态next_state、下一个状态的奖励reward和一个布尔值is_done，表示代理是否已经到达终止状态'''
    def step(self, action): 
        
        tmp_dict = self.netconfig.get(self.cur_state, None) 
        
        if tmp_dict is not None: # 如果 action 在 tmp_dict 中存在，则返回对应的值
            next_state = tmp_dict.get(action, self.pad_idx) # get操作：如果指定的键存在于字典中，则返回对应的值；如果指定的键不存在，则返回提供的默认值（这里是 self.pad_idx[714]）
        else: # ；如果不存在，则返回 self.pad_idx
            next_state = self.pad_idx
        reward = self.get_reward(self.cur_state) # 调用 self.get_reward 方法获取当前状态的奖励 reward
        self.cur_state = next_state # 将当前状态更新为下一个状态 self.cur_state = next_state
        # 判断代理是否已经到达终止状态。如果当前状态等于目标状态self.cur_state == self.cur_des或当前状态等于填充索引self.cur_state == self.pad_idx，则设置done为True，否则设置为False
        done = (self.cur_state == self.cur_des) or (self.cur_state == self.pad_idx)
        return next_state, reward, done # 返回下一个状态next_state、奖励reward、是否终止done

    def get_state_transition(self, state, action):
        return self.netconfig[state][action]

    def get_action_list(self, state): 
        if state in self.netconfig.keys(): 
            return list(self.netconfig[state].keys()) # 返回在当前link（state）下，所有可以执行的下一动作（方向）列表，如[6，2，4]
        else:
            return list() 
    '''从演示数据文件中提取状态、目的地、动作和下一状态'''
    def import_demonstrations(self, demopath, od=None, n_rows=None):
        demo = pd.read_csv(demopath, header=0, nrows=n_rows) # 加载数据
        expert_st, expert_des, expert_ac, expert_st_next = [], [], [], []
        for demo_str, demo_des in zip(demo['path'].tolist(), demo['des'].tolist()): 
            cur_demo = [int(r) for r in demo_str.split('_')] 
            len_demo = len(cur_demo) 
            for i0 in range(1, len_demo): 
                cur_state = cur_demo[i0 - 1] # 
                next_state = cur_demo[i0]    # 
                action_list = self.get_action_list(cur_state) # 
                j = [self.get_state_transition(cur_state, a0) for a0 in action_list].index(next_state) 
                action = action_list[j] # 得到真实的动作
                expert_st.append(cur_state)
                expert_des.append(demo_des)
                expert_ac.append(action)
                expert_st_next.append(next_state)
        return torch.LongTensor(expert_st), torch.LongTensor(expert_des), torch.LongTensor(expert_ac), torch.LongTensor(
            expert_st_next) 

    def import_demonstrations_step(self, demopath, n_rows=None): 
        demo = pd.read_csv(demopath, header=0, nrows=n_rows)
        trajs = []
        for demo_str, demo_des in zip(demo['path'].tolist(), demo['des'].tolist()): 
            cur_demo = [int(r) for r in demo_str.split('_')] 
            len_demo = len(cur_demo)
            episode = []
            for i0 in range(1, len_demo):
                cur_state = cur_demo[i0 - 1] 
                next_state = cur_demo[i0]    

                action_list = self.get_action_list(cur_state) # 返回在当前link（state）下，所有可以执行的下一动作（方向）列表，如[6，2，4]
                j = [self.get_state_transition(cur_state, a0) for a0 in action_list].index(next_state)
                action = action_list[j]

                reward = self.get_reward(cur_state)
                is_done = next_state == demo_des

                episode.append(
                    Step(cur_state=cur_state, action=action, next_state=next_state, reward=reward, done=is_done))
            trajs.append(episode)
            self.max_route_length = len(episode) if self.max_route_length < len(episode) else self.max_route_length
        return trajs