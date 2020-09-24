import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from memory import BatchMaker
from config import gamma, lambda_gae, epsilon_clip, critic_coefficient, entropy_coefficient, epoch_k, batch_size

import warnings


class PPO(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PPO, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def get_gae(self, values, rewards, masks):
        """
        GAE (generalized advantage estimation)
        １：軌跡から各タイムステップでのリターンを計算する
        ２：各タイムステップでのTD誤差を計算する（1ステップ先のみを見た場合のアドバンテージ関数と等しい）
        ３：各タイムステップでのTD誤差に、次のタイムステップでのアドバンテージを減衰率付きで足す
        　　（減衰率γ、1エピソード終了まで延々と先を見るアドバンテージ関数に等しい）

        出力
        各タイムステップのリターン、各タイムステップでの (generalized) Advantage
        """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        previous_value = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            running_advantage = running_tderror + (gamma * lambda_gae) * running_advantage * masks[t]

            returns[t] = running_return
            previous_value = values.data[t]
            advantages[t] = running_advantage

        return returns, advantages

    @classmethod
    def train_model(cls, net, transitions, optimizer):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

        ### いつもの
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        ### まず更新前の policy, value を推論
        ### old_policies：更新時にpolicyの変化率を計算し、その変動をclipするために用いる
        ### old_values  ：学習に用いるadvantageを予め算出するのに用いる
        old_policies, old_values = net(states)
        old_policies = old_policies.view(-1, net.num_outputs).detach()

        ### return：Criticの更新として、Valueをreturnに近づける
        ### advantage：Actorの更新として、Advantageをプラスにする
        returns, advantages = net.get_gae(old_values.view(-1).detach(), rewards, masks)

        batch_maker = BatchMaker(states, actions, returns, advantages, old_policies)
        for _ in range(epoch_k):
            for _ in range(len(states) // batch_size):
                states_sample, actions_sample, returns_sample, advantages_sample, old_policies_sample = batch_maker.sample()

                policies, values = net(states_sample)
                values = values.view(-1)
                policies = policies.view(-1, net.num_outputs)

                ### Actor
                ### Advantageを大きくするように更新する。よって　loss = -Advantage
                ### ただし policies / old_policies_sample が１から離れすぎている場合はクリップする
                ratios = ((policies / old_policies_sample) * actions_sample.detach()).sum(dim=1)
                clipped_ratios = torch.clamp(ratios, min=1.0-epsilon_clip, max=1.0+epsilon_clip)
                actor_loss = -torch.min(ratios * advantages_sample, clipped_ratios * advantages_sample).sum()

                ### Critic
                ### Valueがリターンに近づくように更新する
                critic_loss = (returns_sample.detach() - values).pow(2).sum()

                ### Entropy
                policy_entropy = (torch.log(policies) * policies).sum(1, keepdim=True).mean()

                ### Loss合計
                loss = actor_loss + critic_coefficient * critic_loss - entropy_coefficient * policy_entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss, actor_loss.item(), critic_loss.item()

    def get_action(self, input):
        policy, _ = self.forward(input)

        policy = policy[0].data.numpy()
        action = np.random.choice(self.num_outputs, 1, p=policy)[0]

        return action
