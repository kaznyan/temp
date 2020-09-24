# -*- coding:utf-8 -*-
import argparse
import time

import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import gym
from gym import envs
from gym.envs.registration import register

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

def makeModel(n_act, n_obs):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + n_obs))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(n_act))
    model.add(Activation('linear'))
    print(model.summary())
    return model

parser = argparse.ArgumentParser()
parser.add_argument('mode', help='mode')
parser.add_argument('-save', help='model_name')
parser.add_argument('-load', help='model_name')
args = parser.parse_args()
if args.mode == "train":
    F_train = True
elif args.mode == "test":
    F_train = False
else:
    raise Exception("モードを指定する")


### 環境の指定
# env_name = 'MountainCar-v0'
env_name = 'MyMountainCar-v0'

### 自作環境を呼び出す場合のおまじない
register(id = env_name,
         entry_point = 'MyMountainCarEnv:MyMountainCarEnv')

### デフォルトの環境でも自作環境でも共通
env = gym.make(env_name)

### とりあえず定義する
if args.save:
    save_model_name = args.save
else:
    save_model_name = 'dqn_{}.h5df'.format(env_name)
if args.load:
    load_model_name = args.load
else:
    load_model_name = 'dqn_{}.h5df'.format(env_name)

### --- DQN の NW を組み立てる ---
n_act = env.action_space.n
n_obs = env.observation_space.shape

model = makeModel(n_act, n_obs)
if not F_train:
    model.load_weights(load_model_name)

memory = SequentialMemory(limit=50000, window_length=1)

# policy = BoltzmannQPolicy()
policy = EpsGreedyQPolicy()

dqn = DQNAgent(model = model, 
               nb_actions = n_act, 
               memory = memory, 
               nb_steps_warmup = 10,
               target_model_update = 1e-2, 
               policy = policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

### --- 学習 ---
if F_train:
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights(save_model_name, overwrite=True)

    dqn.test(env, nb_episodes=5, visualize=True)

else:
    dqn.test(env, nb_episodes=5, visualize=True)


