# -*- coding:utf-8 -*-
import time

import numpy as np
import pandas as pd

import gym
from gym import envs
from gym.envs.registration import register

def toDiscrete(obs):
    n_discrete = 10
    discrete_obs = [x.toDiscrete(n_discrete) for x in obs]
    return discrete_obs

def updateQTable(q_table, action, obs, next_obs, reward, episode):
    alpha = 0.2 # 学習率
    gamma = 0.99 # 時間割引き率

    ### 行動後の状態で得られる最大行動価値 max(Q(s',a'))
    sd = toDiscrete(next_obs)
    next_max_q_value = max(q_table[sd[0], sd[1], sd[2], sd[3]])

    ### 行動前の状態の行動価値 Q(s,a)
    s = toDiscrete(obs)
    q_value = q_table[s[0], s[1], s[2], s[3], action]

    ### ベルマン方程式により行動価値関数を更新
    q_table[s[0], s[1], s[2], s[3], action] = q_value + alpha * (reward + gamma * next_max_q_value - q_value)

    return q_table

def qTableAction(env, q_table, obs, episode, F_train=True):
    if F_train:
        epsilon = 0.002
    else:
        epsilon = 0
    if np.random.uniform(0, 1) > epsilon:
        s = toDiscrete(obs)
        qs = q_table[s[0], s[1], s[2], s[3]]
        ### argmax ではなく候補を絞ってそこからランダム　序盤に変な振動をしなくなる（気がする）
        action_cand = np.where(qs == qs.max())[0]
        action = action_cand[np.random.randint(0, action_cand.size)]
    else:
        action = np.random.choice([0, 3])
    return action

### train か test か
F_train = True

### 環境の指定
### A. 自作環境を呼び出す場合のおまじない
register(
    id='MyCatchEnv-v0',
    entry_point='MyCatchEnv_ra:MyEnv'
)
env = gym.make('MyCatchEnv-v0')
### B. 既定の環境を呼び出す場合
# env = gym.make('MountainCar-v0')

### Q学習用
q_table = np.zeros((10, 10, 10, 10, 9))

num_iter = 10000
df_stats = pd.DataFrame(index=list(range(num_iter)), columns=["Flag"])

### --- ここから処理 ---
### 今回のobs = [物体のx座標, ロボットのx座標]
### 今回のaction = [0, 1, 2]
episode = 0

for i_loop in range(num_iter):
    if i_loop % 100 == 0:
        print(i_loop)
        df_stats.to_csv("aaa.csv")
    action = 0
    total_reward = 0
    obs = env.reset()

    for i in range(400):
        ### 次のアクションを決定
        action = qTableAction(env, q_table, obs, episode, F_train=True)

        ### actionを起こし結果を受け取る
        next_obs, reward, done, info = env.step(action)

        q_table = updateQTable(q_table, action, obs, next_obs, reward, episode)
        obs = next_obs

        ### 終了判定
        if done:
            if i < 399:
                df_stats.loc[i_loop, "Flag"] = 1
                print([x.val for x in next_obs])
            break


    else: ### 終了判定がなかった場合
        df_stats.loc[i_loop, "Flag"] = 0


### --- 学習後に1回だけ可視化をする
df = pd.DataFrame(index=range(400), columns=["a", "b", "c", "d", "e", "f"])
obs = env.reset()
df.loc[0] = np.asarray([x.val for x in obs]).reshape((1, -1))
for i in range(400):
    ### 次のアクションを決定
    action = qTableAction(env, q_table, obs, episode, F_train=False)

    ### actionを起こし結果を受け取る
    next_obs, reward, done, info = env.step(action)

    q_table = updateQTable(q_table, action, obs, next_obs, reward, episode)
    obs = next_obs

    ### 終了判定
    if done:
        if i < 399:
            df_stats.loc[i_loop, "Flag"] = 1
            print([x.val for x in next_obs])
        break

    df.loc[i] = np.asarray([x.val for x in obs]).reshape((1, -1))

df.to_csv("bbb.csv")



#
