# -*- coding: utf-8 -*-

import os
import sys
import gym
import random
import numpy as np
# import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet

from config import env_name, goal_score, log_interval, device, lr
from memory import Memory

def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    ### ポリシーネットワークの構築
    ### inputに対してπ(a|s) と V(s) が出力される
    ### Vの出力は1つ　が学習時にはAdvantage関数を計算する
    net = QNet(num_inputs, num_actions)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    net.to(device)
    net.train()

    ### もろもろの初期化
    running_score = 0
    steps = 0
    loss = 0
    steps_before = 0

    df = pd.DataFrame(index=range(10000), columns=["steps", "loss_policy", "loss_value"])

    for e in range(10000):
        done = False
        ### 1エピソード分のメモリすら持たずに1ステップずつ学習

        ### 環境を初期状態に
        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        memory = Memory()

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

            transition = [state, next_state, action, reward, mask]

            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

        ### 1エピソードの結果を学習
        loss, loss_policy, loss_value = QNet.train_model(net, optimizer, memory.sample())
        print("Ep {0:04d}: {1} step, loss_policy: {2}, loss_value: {3}".format(e, steps - steps_before, loss_policy, loss_value))
        df.loc[e, "steps"]       = steps - steps_before
        df.loc[e, "loss_policy"] = loss_policy
        df.loc[e, "loss_value"]  = loss_value
        steps_before = steps

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f}'.format(e, running_score))

        if running_score > goal_score:
            break
    df.to_csv("loss.csv")


if __name__=="__main__":
    main()
