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
        self.fc_critic = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = F.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x))
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def train_model(cls, net, optimizer, memory_list):
        batch_list = [x.sample() for x in memory_list]

        states_list = []
        next_states_list = []
        actions_list = []
        # rewards_list = []
        returns_list = []
        for batch in batch_list:
            states      = torch.stack(batch.state).view(-1, net.num_inputs)
            next_states = torch.stack(batch.next_state)
            actions     = torch.stack(batch.action)
            rewards     = torch.Tensor(batch.reward)
            masks       = torch.Tensor(batch.mask)

            ### 1epの軌跡から厳密なrewardを求められる
            _, last_value = net(next_states[-1]) ### 最後の状態の価値だけ評価する
            temp_return = last_value[0].data
            returns = torch.zeros(rewards.size())
            for t in reversed(range(0, len(rewards))):
                temp_return = rewards[t] + gamma * temp_return * masks[t]
                returns[t] = temp_return

            ### concatするための一時置き場
            states_list.append(states)
            next_states_list.append(next_states)
            actions_list.append(actions)
            returns_list.append(returns)

        ### concatして学習の準備完了
        states      = torch.cat(states_list, dim=0)
        next_states = torch.cat(next_states_list, dim=0)
        actions     = torch.cat(actions_list, dim=0)
        returns     = torch.cat(returns_list, dim=0)

        ### 満を持して、まとめて評価する
        policy, value = net(states)
        policy = policy.view(-1, net.num_outputs)
        value  = value.view(-1)

        ### 価値関数Vの更新：TD誤差を0に近づけること
        td_error = returns - value.detach()
        loss_value = torch.pow(td_error, 2).view(-1, 1)

        ### 方策の更新：方策勾配法っぽいもの　ベースラインとしてVを用いる
        log_policy = (torch.log(policy + 1e-10) * actions).sum(dim=1, keepdim=True)
        loss_policy = - log_policy.view(-1, 1) * td_error.view(-1, 1)

        ### policyのentropyを大きくすることで探索を促進する
        entropy = (torch.log(policy + 1e-10) * policy).sum(dim=1, keepdim=True).view(-1, 1)

        ### 2つのLossを足してentropyを引いたものをlossとする
        ### 重みが1:1:0.1 なのはあまりに適当な気がするが……
        loss = (loss_policy + loss_value - 0.1 * entropy).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, loss_policy.mean().item(), loss_value.mean().item()
        # return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        policy = policy[0].data.numpy()

        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action
