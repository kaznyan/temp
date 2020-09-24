# -*- coding: utf-8 -*-
import gym
import numpy as np
import time

class Agent:
    def __init__(self, num_s, num_a):
        self.num_s = num_s
        self.num_a = num_a
        self.brain = Brain(num_s, num_a)

    def update_q_function(self, s, a, reward, next_s):
        self.brain.update_Qtable(s, a, reward, next_s)

    def get_action(self, s, step):
        a = self.brain.decide_action(s, step)
        return a

    def get_action_test(self, s):
        a = self.brain.decide_action_test(s)
        return a

class Brain:
    def __init__(self, num_s, num_a):
        self.num_s = num_s
        self.num_a = num_a
        self.q_table = np.random.uniform(low=0, high=1, size=(NUM_DIZITIZED ** self.num_s, self.num_a))

    def bins(self, clip_min, clip_max, num):
        #観測した状態（連続値）を離散値にデジタル変換する
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, s):
        # 観測した状態を、離散値に変換する
        cart_pos, cart_v, pole_angle, pole_v = s
        digitized = [
                    np.digitize(cart_pos,   bins=self.bins(-2.4, 2.4, NUM_DIZITIZED)),
                    np.digitize(cart_v,     bins=self.bins(-3.0, 3.0, NUM_DIZITIZED)),
                    np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZED)),
                    np.digitize(pole_v,     bins=self.bins(-2.0, 2.0, NUM_DIZITIZED))
                    ]
        return sum([x * (NUM_DIZITIZED ** i) for i, x in enumerate(digitized)])

    def update_Qtable(self, s, a, reward, next_s):
        s = self.digitize_state(s)
        next_s = self.digitize_state(next_s)
        Max_Q_next = max(self.q_table[next_s][:])
        self.q_table[s, a] = self.q_table[s, a] + ETA * (reward + GAMMA * Max_Q_next - self.q_table[s, a])

    def decide_action(self, s, episode):
        s = self.digitize_state(s)
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            a = np.argmax(self.q_table[s][:])
        else:
            a = np.random.choice(self.num_a) # 0, 1の行動をランダムに返す
        return a

    def decide_action_test(self, s):
        s = self.digitize_state(s)
        a = np.argmax(self.q_table[s][:])
        return a


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        self.num_s = self.env.observation_space.shape[0]
        self.num_a = self.env.action_space.n
        self.agent = Agent(self.num_s, self.num_a)

    def run(self):
        complete_episodes = 0 ### 195step 以上連続で立ち続けた試行数
        for episode in range(NUM_EPISODES):
            s = self.env.reset()
            episode_reward = 0

            for step in range(MAX_STEPS):
                a = self.agent.get_action(s, episode)

                next_s, reward_notuse, done, _ = self.env.step(a)

                if done: # ステップ数経過、または一定角度以上傾くとtrue
                    if step < MAX_STEPS - 5:
                        reward = -1 ### 途中でこけたら報酬 = -1
                        self.complete_episodes = 0
                    else:
                        reward = 1  # 立ったまま終了したら報酬 = 1
                        self.complete_episodes = self.complete_episodes + 1
                else:
                    reward = 0 ### 何事もなかったら報酬 = 0
                episode_reward += reward  # 報酬を追加

                self.agent.update_q_function(s, a, reward, next_s)
                s = next_s

                if done:
                    print('{0} Episode: Finished after {1} time steps'.format(episode, step + 1))
                    break

            if self.complete_episodes >= 10:
                print('10回連続成功')
                break

    def run_test(self):
        s = self.env.reset()
        self.env.render()

        for step in range(MAX_STEPS):
            print(step)
            a = self.agent.get_action_test(s)
            s, reward_notuse, done, _ = self.env.step(a)
            self.env.render()
            if done:
                time.sleep(1)
                break


ENV = 'CartPole-v0'
NUM_DIZITIZED = 6  # 各状態の離散値への分割数
GAMMA = 0.99  # 時間割引率
ETA = 0.5  # 学習係数
MAX_STEPS = 200  # 1試行のstep数
NUM_EPISODES = 1000  # 最大試行回数

# main クラス
cartpole_env = Environment()
cartpole_env.run()
cartpole_env.run_test()










#
