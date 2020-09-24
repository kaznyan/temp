import os
import sys
import gym
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet

from memory import Memory
from config import env_name, goal_score, log_interval, device, lr, gamma


def main():
    ### 環境を初期化
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    ### ポリシーネットワークの構築
    net = QNet(num_inputs, num_actions)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    net.to(device)
    net.train()

    ### もろもろの初期化
    running_score = 0
    steps = 0
    loss = 0
    steps_before = 0

    for e in range(10000):
        done = False
        ### 1エピソードごとにMemoryは空にする（実質、Experience Replay がない）
        memory = Memory()

        ### 環境を初期状態に
        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            ### epsilon は使わず、各行動の評価値を確率に直接変換して行動を決定
            action = net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            action_one_hot = torch.zeros(num_actions)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

        ### 1エピソード分をまとめて学習
        ### memory.sample はランダムに選択ではなく、1エピソードのmemory全体を返す
        loss = QNet.train_model(net, optimizer, memory.sample())

        print("Ep {0:04d}: {1} step".format(e, steps - steps_before))
        steps_before = steps

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f}'.format(e, running_score))

        if running_score > goal_score:
            break


if __name__=="__main__":
    main()
