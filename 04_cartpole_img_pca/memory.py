import os, shutil
import csv
import random
from collections import namedtuple, deque

import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Memory(object):
    def __init__(self, capacity, memory_dir, n_states):
        # self.memory = deque(maxlen=capacity)
        self.capacity = capacity
        self.memory_dir = memory_dir
        self.n_states = n_states
        if os.path.exists(self.memory_dir):
            shutil.rmtree(self.memory_dir)
        os.mkdir(self.memory_dir)
        self.count = 0

    def push(self, state, next_state, action, reward, mask):
        ### dequeの機能にappendなのでオーバーしたら最初のほうから消えていく
        # self.memory.append(Transition(state, next_state, action, reward, mask))
        state      = self._tensor_to_csv(state)
        next_state = self._tensor_to_csv(next_state)
        action     = self._np_to_csv(action)
        reward     = str(reward)
        mask       = str(mask)
        txt = state + ",next_state," + next_state + ",action," + action + ",reward," + reward + ",mask," + mask
        with open(self.memory_dir + "{0:04d}.csv".format(self.count), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(txt.split(","))
        self.count += 1
        if self.count >= self.capacity:
            self.count = 0

    def _tensor_to_csv(self, tensor):
        tensor = tensor.detach().numpy().reshape((-1)).tolist()
        tensor = [str(x) for x in tensor]
        tensor = ",".join(tensor)
        return tensor

    def _np_to_csv(self, array):
        array = array.reshape((-1)).tolist()
        array = [str(x) for x in array]
        array = ",".join(array)
        return array

    def sample(self, batch_size):
        # transitions = random.sample(self.memory, batch_size)
        # batch = Transition(*zip(*transitions))
        rand_list = np.random.randint(0, self.capacity, [batch_size])

        batch = {}
        state_list      = np.zeros((batch_size, self.n_states))
        next_state_list = np.zeros((batch_size, self.n_states))
        action_list     = np.zeros((batch_size, 2))
        reward_list     = np.zeros((batch_size, 1))
        mask_list       = np.zeros((batch_size, 1))

        for i, rand in enumerate(rand_list):
            with open(self.memory_dir + "{0:04d}.csv".format(rand)) as f:
                txt = f.read()
            state, txt      = txt.split(",next_state,")
            next_state, txt = txt.split(",action,")
            action, txt     = txt.split(",reward,")
            reward, mask    = txt.split(",mask,")
            state      = state.split(",")
            next_state = next_state.split(",")
            action     = action.split(",")
            reward     = reward.split(",")
            mask       = mask.split(",")
            state_list[i]      = np.asarray(state)
            next_state_list[i] = np.asarray(next_state)
            action_list[i]     = np.asarray(action)
            reward_list[i]     = np.asarray(reward)
            mask_list[i]       = np.asarray(mask)
        batch["state"]      = torch.from_numpy(state_list.astype(np.float32))
        batch["next_state"] = torch.from_numpy(next_state_list.astype(np.float32))
        batch["action"]     = torch.from_numpy(action_list.astype(np.float32))
        batch["reward"]     = torch.from_numpy(reward_list.astype(np.float32).reshape((-1)))
        batch["mask"]       = torch.from_numpy(mask_list.astype(np.float32).reshape((-1)))

        return batch

    def __len__(self):
        return len(self.memory)
