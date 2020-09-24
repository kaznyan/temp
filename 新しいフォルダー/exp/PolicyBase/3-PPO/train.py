import os
import sys
import gym
import random
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import PPO

from memory import Memory
from config import env_name, goal_score, log_interval, device, gamma, lr


def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = PPO(num_inputs, num_actions)

    optimizer = optim.Adam(net.parameters(), lr=lr)

    net.to(device)
    net.train()
    running_score = 0
    steps = 0
    loss = 0
    steps_before = 0

    df = pd.DataFrame(index=range(10000), columns=["steps", "loss_policy", "loss_value"])

    for e in range(30000):
        done = False
        memory = Memory()

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            action_one_hot = torch.zeros(2)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

        loss, loss_policy, loss_value = PPO.train_model(net, memory.sample(), optimizer)

        score = score if score == 500.0 else score + 1
        if running_score == 0:
            running_score = score
        running_score = 0.99 * running_score + 0.01 * score
        print("Ep {0:04d}: {1} step, loss_policy: {2}, loss_value: {3}".format(e, steps - steps_before, loss_policy, loss_value))
        df.loc[e, "steps"]       = steps - steps_before
        df.loc[e, "loss_policy"] = loss_policy
        df.loc[e, "loss_value"]  = loss_value
        steps_before = steps

        if e % log_interval == 0:
            print('{} episode | score: {:.2f}'.format(e, running_score))

        if running_score > goal_score:
            break
    df.to_csv("loss.csv")


if __name__=="__main__":
    main()
