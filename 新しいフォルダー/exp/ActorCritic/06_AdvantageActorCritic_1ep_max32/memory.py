import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

class Memory(object):
    def __init__(self):
        self.memory = deque()

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self):
        memory = self.memory

        batch_size = min(len(memory), 32)
        memory = random.sample(memory, batch_size)

        return Transition(*zip(*memory))

    def __len__(self):
        return len(self.memory)
