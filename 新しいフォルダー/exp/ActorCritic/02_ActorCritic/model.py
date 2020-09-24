import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import gamma
class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x))
        q_value = self.fc_critic(x)
        return policy, q_value

    @classmethod
    def train_model(cls, net, optimizer, transition):
        state, next_state, action, reward, mask = transition

        policy, q_value = net(state)
        policy, q_value = policy.view(-1, net.num_outputs), q_value.view(-1, net.num_outputs) ### 成型のみ
        _, next_q_value = net(next_state)
        next_q_value = next_q_value.view(-1, net.num_outputs) ### 成型のみ
        next_action = net.get_action(next_state)

        ### Q値をヒントに用いて方策が良くなるように導く
        ### Q値が大きい行動の確率が低いとLossが大きくなる
        log_policy = torch.log(policy[0])[action]
        loss_policy = - log_policy * q_value[0][action].item()

        ### Q関数の更新：SARSA風のベルマン更新
        target = reward + mask * gamma * next_q_value[0][next_action]
        loss_value = F.mse_loss(q_value[0][action], target.detach())

        ### 2つのLossを足して更新する
        ### 重みが1:1なのはあまりに適当な気がするが……
        loss = loss_policy + loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, loss_policy, loss_value
        # return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
