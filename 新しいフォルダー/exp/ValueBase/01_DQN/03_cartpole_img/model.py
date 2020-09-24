import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch):
        ### 経験データの取り出し
        # states = torch.stack(batch.state)
        # next_states = torch.stack(batch.next_state)
        # actions = torch.Tensor(batch.action).float()
        # rewards = torch.Tensor(batch.reward)
        # masks = torch.Tensor(batch.mask)
        states      = batch["state"]
        next_states = batch["next_state"]
        actions     = batch["action"]
        rewards     = batch["reward"]
        masks       = batch["mask"]

        ### 状態sにおけるQ(s, a1), Q(s, a2), ... を推論により求める × バッチ数
        pred = online_net(states).squeeze(1)
        ### 状態s'におけるQ(s', a1), Q(s', a2), ... を推論により求める × バッチ数
        next_pred = target_net(next_states).squeeze(1)

        ### Q(s, a)を求める　actionsはone-hotになっている
        pred = torch.sum(pred.mul(actions), dim=1)

        ### rt + γ max(Q(s', a))
        target = rewards + masks * gamma * next_pred.max(1)[0]

        ### Q学習の更新式は
        ###     Q(st, at) := Q(st, at) + αtδt　※αtは学習率、δtはTD誤差
        ###     δt = rt + γ max(Q(s(t+1), a')) - Q(st, at)
        ### 書き換えると
        ###     Q(st, at) := (1-αt) Q(st, at) + αt (rt + γ max(Q(s(t+1), a')))
        ### これの解釈として、
        ###     Q(st, at) を rt + γ max(Q(s(t+1), a')) に少し近づける
        ### ということになるので、
        ###     Q値を推論するNNのtargetを rt + γ max(Q(s(t+1), a')) とすることでうまく学習できる
        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]
