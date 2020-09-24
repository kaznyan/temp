import math
import random
import copy

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

class MyEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity = 0):
        ### 将来的には速度とかも含む
        self.vel = 0

        ### 甲斐流：今回のS
        self.x_obj = StateVariable(-2, 2)
        self.y_obj = StateVariable(-2, 2)
        self.x_rob = StateVariable(-2, 2, fix_init_val = 0)
        self.y_rob = StateVariable(-2, 2, fix_init_val = 0)

    def step(self, action):
        ### actionによる環境の変化
        self._changeState(action)
        ### それによる終了判定
        F_done = self._checkDone()
        ### そのときの報酬
        reward = self._calcReward()
        ### 観測　NOTE: list にすることで扱いやすくしているが deepcopy にしないと step のときに旧obs が書き換わる
        obs = copy.deepcopy([self.x_obj, self.y_obj, self.x_rob, self.y_rob])
        return obs, reward, F_done, {}

    def _changeState(self, action): ### 環境のルールによって変える
        x_move = (action // 2) * 2 - 1 ### -1 か +1
        y_move = (action %  2) * 2 - 1 ### -1 か +1
        self.x_rob.gain(x_move * 0.05)
        self.y_rob.gain(y_move * 0.05)

    def _checkDone(self): ### 環境のルールによって変える
        F_1 = (abs(self.x_obj.val - self.x_rob.val) < 0.1)
        F_2 = (abs(self.y_obj.val - self.y_rob.val) < 0.1)
        done = (F_1 and F_2)
        return done

    def _calcReward(self):
        x_distance = abs(self.x_obj.val - self.x_rob.val)
        y_distance = abs(self.y_obj.val - self.y_rob.val)
        reward = -(x_distance + y_distance) ### 負の数にする
        return reward

    def reset(self):
        state_list = copy.deepcopy([self.x_obj.initValue(),
                                    self.y_obj.initValue(),
                                    self.x_rob.initValue(),
                                    self.y_rob.initValue()])
        return state_list


class StateVariable(object):
    def __init__(self, min_value, max_value, fix_init_val=None):
        self.min = min_value
        self.max = max_value
        if fix_init_val is None:
            self.F_fix = False
            self.init_val = None
        else:
            self.F_fix = True
            self.init_val = fix_init_val

    def initValue(self):
        if self.F_fix:
            self.val = self.init_val
        else:
            self.val = random.uniform(self.min, self.max)
        self.clip()
        return self

    def toDiscrete(self, n):
        dx = (self.max - self.min) / n
        discrete = int((self.val - self.min) / dx)
        discrete = min(n - 1, discrete) ### 万一 max と一致すると index out of range なので
        return discrete

    def clip(self):
        self.val = np.clip(self.val, self.min, self.max)

    def gain(self, gain):
        self.val += gain
        self.clip()


#
