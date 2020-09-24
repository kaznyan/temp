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
from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr


def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)

def update_target_model(online_net, target_net):
    # target_net を online_net と同じものにする
    target_net.load_state_dict(online_net.state_dict())


def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    ### NNのIn-Outは環境によって異なる
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    ### 2つのNWを作成・初期化
    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    ### 各NWの設定　CPU / GPU
    online_net.to(device)
    target_net.to(device)
    ### 各NWの設定　初めは学習モードにする
    online_net.train()
    target_net.train()

    ### 学習前の初期設定
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0
    steps_before = 0

    for e in range(3000):
        done = False

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            ### 行動の決定はtarget_netで行う
            action = get_action(state, target_net, epsilon, env)

            ### 次の状態の観測、報酬の獲得
            next_state, reward, done, _ = env.step(action)
            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            ### わかりにくいので書き変えた
            if done:
                mask = 0
            else:
                mask = 1
            if (done and (score != 499)): ### 499ステップまで行かずにdoneになったら
                reward = -1
            else:
                pass ### rewardは基本的に1
            # mask = 0 if done else 1
            # reward = reward if not done or score == 499 else -1

            ### memoryに記録
            action_one_hot = np.zeros(2)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            ### rewardは基本的に1
            score += reward ### そのepisodeで何ステップ行ったかを記録するためだけのもの

            state = next_state

            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                ### online_net の学習
                batch = memory.sample(batch_size)
                loss = QNet.train_model(online_net, target_net, optimizer, batch)

                ### たまにtarget_netをonline_netで上書きする
                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        print("Ep {0:04d}: {1} step".format(e, steps - steps_before))
        steps_before = steps

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(e, running_score, epsilon))

        if running_score > goal_score:
            break


if __name__=="__main__":
    main()
