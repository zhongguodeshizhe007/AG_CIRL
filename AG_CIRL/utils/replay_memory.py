from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb
'''代理通过与环境交互收集转换并将其存储在内存缓冲区中，代理可以稍后从内存缓冲区中采样转换，用于训练神经网络或更新其策略'''

'''
创建具名元组（named tuple）：

state：当前状态。
destination：目标状态。
action：在当前状态下采取的动作。
next_state：采取动作后的下一个状态。
reward：从转换中获得的奖励。
mask：指示转换是否终止的掩码（如果终止，则为0，否则为1）。
bad_mask：指示下一个状态是否为坏状态的掩码（如果是坏状态，则为0，否则为1）。
'''
Transition = namedtuple('Transition', ('state', 'destination', 'action', 'next_state', 'reward', 'mask', 'bad_mask')) #

'''
__init__()：初始化一个空的内存缓冲区。
push()：通过将transition追加到内存缓冲区来保存transition。传递给push()的参数应与Transition的字段对应。
sample()：从内存缓冲区中随机采样transition。如果提供了batch_size，则返回大小为batch_size的一批transition，否则返回缓冲区中的所有转换。返回的转换以Transition的形式呈现，其中每个字段包含一个值列表。
append()：将另一个Memory对象中的transition附加到当前内存缓冲区。
__len__()：返回内存缓冲区中存储的transition数量。
'''

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args)) # Transition(*args)表示将args中的元素作为参数传递给Transition这个namedtuple的构造函数，从而创建一个Transition对象
                                              # args是一个可变长度的参数列表，*args使用了解包操作符*，它的作用是将一个可迭代对象（例如列表、元组）解包成单独的元素
                                              # 其中，args = (state, destination, action, next_state, reward, mask, bad_mask)
    '''
    下列sample()函数作用，Transition(*zip(*self.memory))的作用是将self.memory中的多个Transition对象解压并重新组合成一个新的Transition对象，其中每个字段都包含了原始对象中相同位置的元素。
    1. zip(*self.memory)将self.memory列表中的Transition对象解压，并将相同位置的元素分组。例如，如果self.memory包含两个Transition对象:
    self.memory = [Transition(1, 2, 3), Transition(4, 5, 6)]
    那么zip(*self.memory)将产生一个可迭代对象：
    zip_obj = zip(Transition(1, 2, 3), Transition(4, 5, 6))
    注意，zip函数的作用是将多个可迭代对象的对应元素组合成元组。在这里，它将两个Transition对象的相同位置的元素进行组合
    2. *zip(*self.memory)使用解压操作符*对zip对象进行解压，相当于对每个元组进行解压，将元组中的元素作为参数传递给Transition的构造函数。这将产生以下效果：
    Transition(*zip(*self.memory)) = Transition((1, 4), (2, 5), (3, 6))
    这样就创建了一个新的Transition对象，其中每个字段都包含了原始self.memory中对应位置的元素组成的元组
    '''
    def sample(self, batch_size=None):
        if batch_size is None:
            # self.memory是一个存储Transition对象的列表。Transition(*zip(*self.memory))的作用是将self.memory中的多个Transition对象解压并重新组合成新的Transition对象
            return Transition(*zip(*self.memory)) 
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch)) 

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)