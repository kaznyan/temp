# -*- coding:utf-8 -*-
import time

import numpy as np
import pandas as pd

import gym
from gym import envs
from gym.envs.registration import register

def ruleBaseAction(obs):
    ### 右に進んでいるなら右に押す　左なら左
    if obs[1] > 0:
        action = 2
    else:
        action = 0
    return action

def toDiscrete(obs):
    n_discrete = 20
    ### 最大・最小
    low  = env.observation_space.low
    high = env.observation_space.high
    ### 幅
    pos_dx, vel_dx = (high - low) / n_discrete
    ### 離散値に変換
    pos = int((obs[0] - low[0]) / pos_dx)
    vel = int((obs[1] - low[1]) / vel_dx)
    return pos, vel

def updateQTable(q_table, action, obs, next_obs, reward, episode):
    alpha = 0.2 # 学習率
    gamma = 0.99 # 時間割引き率

    # 行動後の状態で得られる最大行動価値 Q(s',a')
    next_pos, next_vel = toDiscrete(next_obs)
    next_max_q_value = max(q_table[next_pos][next_vel])

    # 行動前の状態の行動価値 Q(s,a)
    pos, vel = toDiscrete(obs)
    q_value = q_table[pos][vel][action]

    # 行動価値関数の更新
    q_table[pos][vel][action] = q_value + alpha * (reward + gamma * next_max_q_value - q_value)

    return q_table

def qTableAction(env, q_table, obs, episode, F_train=True):
    if F_train:
        epsilon = 0.002
    else:
        epsilon = 0
    if np.random.uniform(0, 1) > epsilon:
        pos, vel = toDiscrete(obs)
        action = np.argmax(q_table[pos][vel])
    else:
        action = np.random.choice([0, 1, 2])
    return action

### train か test か
F_train = True

### 環境の指定
### A. 自作環境を呼び出す場合のおまじない
register(
    id='MyMountainCar-v0',
    entry_point='MyMountainCarEnv:MyMountainCarEnv'
)
env = gym.make('MyMountainCar-v0')
### B. 既定の環境を呼び出す場合
# env = gym.make('MountainCar-v0')

### Q学習用
q_table = np.zeros((20, 20, 3))


if F_train:
    ### 今回のobs = [x座標, x方向への速度]
    ### 今回のaction = [0, 1, 2]
    episode = 0

    for i_loop in range(10000):
        if i_loop % 100 == 0:
            print(i_loop)
        action = 0
        total_reward = 0
        obs = env.reset()

        # ### 録画用
        # env.render()
        # time.sleep(10)

        ### 最初のアクションを決定する
        action = qTableAction(env, q_table, obs, episode, F_train=F_train)

        for i in range(400):
            ### actionを起こし結果を受け取る
            next_obs, reward, done, info = env.step(action)

            q_table = updateQTable(q_table, action, obs, next_obs, reward, episode)

            ### 現在の環境を描画
            # env.render()

            ### 終了判定
            if done:
                # if i < 199:
                #     print(i_loop)
                break

            ### 次のアクションを決定
            # action = ruleBaseAction(obs)
            action = qTableAction(env, q_table, obs, episode, F_train=F_train)

            obs = next_obs

    df = pd.DataFrame(q_table[:, :, 0])
    df.to_csv("action_0.csv", index=False)
    df = pd.DataFrame(q_table[:, :, 1])
    df.to_csv("action_1.csv", index=False)
    df = pd.DataFrame(q_table[:, :, 2])
    df.to_csv("action_2.csv", index=False)

else:
    q_table[:, :, 0] = np.asarray(pd.read_csv("action_0.csv"))
    q_table[:, :, 1] = np.asarray(pd.read_csv("action_1.csv"))
    q_table[:, :, 2] = np.asarray(pd.read_csv("action_2.csv"))
    episode = 0

    action = 0
    obs = env.reset()

    ### 録画用
    # env.render()
    # time.sleep(10)

    ### 最初のアクションを決定する
    action = qTableAction(env, q_table, obs, episode, F_train=F_train)

    for i in range(200):
        ### actionを起こし結果を受け取る
        next_obs, reward, done, info = env.step(action)

        ### 現在の環境を描画
        env.render()

        ### 終了判定
        if done:
            print(i)
            break

        ### 次のアクションを決定
        # action = ruleBaseAction(obs)
        action = qTableAction(env, q_table, obs, episode, F_train=F_train)

        obs = next_obs
