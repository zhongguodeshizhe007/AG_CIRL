import torch
from utils.torch import to_device

def estimate_advantages(rewards, masks, bad_masks, values, next_values, gamma, tau, device):
    # 将rewards、masks、bad_masks、values和next_values转移到CPU设备上
    rewards, masks, bad_masks, values, next_values = to_device(torch.device('cpu'),
                                                               rewards, masks, bad_masks,
                                                               values, next_values)
    tensor_type = type(rewards) # 定义一个变量tensor_type，它存储了rewards张量的类型
    # 下面两行代码创建了形状为(rewards.size(0), 1)的空张量deltas和advantages，其类型与rewards张量相同。
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)
    # prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * next_values[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        advantages[i] = advantages[i] * bad_masks[i]

        # prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
    # 下面这两行代码计算回报值returns和标准化后的advantages。回报值返回values和advantages的和。advantages通过减去均值并除以标准差进行标准化
    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns) # 将advantages和returns转移到指定的设备上，这个设备由变量device指定
    return advantages, returns