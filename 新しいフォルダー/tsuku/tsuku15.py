# -*- coding: utf-8 -*-
from collections import namedtuple
import random
import math
import time

import gym
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity: ### 空の部分を埋める
            self.memory.append(None)
        # else:
        #     self.index = np.random.randint(self.capacity)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        self.init_memory()
        self.init_model()

    def init_memory(self):
        self.memory = ReplayMemory(CAPACITY)

    def init_model(self):
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(self.num_states, 128))
        self.model.add_module('fc3', nn.Linear(128, self.num_actions))
        # self.model.add_module('fc1', nn.Linear(self.num_states, 32))
        # self.model.add_module('relu1', nn.ReLU())
        # self.model.add_module('fc2', nn.Linear(32, 32))
        # self.model.add_module('relu2', nn.ReLU())
        # self.model.add_module('fc3', nn.Linear(32, self.num_actions))
        # self.model.add_module('sof3', nn.Softmax())
        # 最適化手法の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001)
        # self.optimizer = optim.RMSprop(self.model.parameters())
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        # ミニバッチの作成-----------------

        # transitionsは1stepごとの(state, action, state_next, reward)が、BATCH_SIZE分格納されている
        # つまり、(state, action, state_next, reward)×BATCH_SIZE
        # これをミニバッチにしたい。つまり
        # (state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        # バッチから状態、行動、報酬を格納（non_finalはdoneになっていないstate）
        # catはConcatenates（結合）のことです。
        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がBATCH_SIZE分並んでいるのですが、
        # それを size BATCH_SIZEx4 に変換します
        state_batch  = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]))

        # ミニバッチの作成終了------------------

        # ネットワークを推論モードに切り替え
        self.model.eval()

        # Q(s_t, a_t)を求める
        # self.model(state_batch)は、[torch.FloatTensor of size BATCH_SIZEx2]になっており、
        # 実行したアクションに対応する[torch.FloatTensor of size BATCH_SIZEx1]にするためにgatherを使用
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # max{Q(s_t+1, a)}値を求める
        # 次の状態がない場合は0にしておく
        next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.FloatTensor))

        # 次の状態がある場合の値を求める
        # 出力であるdataにアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてその値（index=0）を出力します
        next_state_values[non_final_mask] = self.model(non_final_next_states).data.max(1)[0]

        # 教師となるQ(s_t, a_t)値を求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # ネットワークを訓練モードに切り替え
        self.model.train()

        # 損失関数を計算　smooth_l1_lossはHuberloss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # ネットワークを更新
        self.optimizer.zero_grad() # 勾配をリセット
        loss.backward() # バックプロパゲーションを計算
        self.optimizer.step() # 結合パラメータを更新

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        # epsilon = 0.5 * (1 / (episode * 0.001 + 1))
        # epsilon = 0.5
        # epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)
        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            action = self.model(Variable(state)).data.max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)はtorch.LongTensor of size 1　を size 1x1 に変換します
        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
            # actionは[torch.LongTensor of size 1x1]の形になります
        return action


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.agent = Agent(self.num_states, self.num_actions)
        self.total_step = np.zeros(10)

    def run(self):
        complete_episodes = 0 # 195step以上連続で立ち続けた試行数
        episode_final = False # 最後の試行フラグ
        for episode in range(NUM_EPISODES):
            state = self.env.reset()
            state = torch.from_numpy(state).type(torch.FloatTensor) # numpy変数をPyTorchのテンソルに変換
            state = torch.unsqueeze(state, 0) # FloatTensorof size 4 → size 1x4に変換

            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode)
                # action = self.agent.get_action(state, step)

                state_next, _, done, _ = self.env.step(int(action[0, 0]))

                if done: # ステップ数経過、または一定角度以上傾くとtrue
                    state_next = None
                    self.total_step = np.hstack((self.total_step[1:], step + 1))
                    if step < MAX_STEPS - 5:
                        reward = torch.FloatTensor([-1.0]) ### 途中でこけたら報酬 = -1
                        self.complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0]) ### 立ったまま終了したら報酬 = 1
                        self.complete_episodes += 1
                else:
                    reward = torch.FloatTensor([0.0]) ### 何事もなかったら報酬 = 0
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor) # numpy変数をPyTorchのテンソルに変換
                    state_next = torch.unsqueeze(state_next, 0) # FloatTensorof size 4 → size 1x4に変換

                self.agent.memorize(state, action, state_next, reward)
                self.agent.update_q_function()
                state = state_next

                if done:
                    print('%d Episode: Finished after %d steps：10Average = %.1lf' % (episode, step + 1, self.total_step.mean()))
                    break
            # self.agent.update_q_function()

            if episode_final is True:
                break

            # 10連続で200step立ち続けたら成功
            if self.complete_episodes >= 10:
                print('10回連続成功')
                episode_final = True # 次の試行を最終試行とする


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
# MAX_STEPS = 50
NUM_EPISODES = 50000

BATCH_SIZE = 32
# BATCH_SIZE = 1024
# CAPACITY = 1024
CAPACITY = 10000
# CAPACITY = 100000000

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

cartpole_env = Environment()
cartpole_env.run()
# cartpole_env.run_test()










#
