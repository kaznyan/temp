import numpy as np
import pandas as pd
import torch

from agents import PPO
from curiosity import ICM, MlpICMModel, NoCuriosity
from envs import MultiEnv
from models import MLP
from reporters import TensorBoardReporter
from rewards import GeneralizedAdvantageEstimation, GeneralizedRewardEstimation

"""
Pendulum環境

state = cos(theta), sin(theta), d(theta)/dt
action = トルク（連続値）
done は判定が終了時のみ

制御対象が
最も上にあるとき、theta = 0
最も下にあるとき、theta = -180 または +180

※thetaを返すと-180度のところで非連続になる
　代わりにsin, cosを返すことで連続になる
※d(theta)/dt は角速度
"""

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reporter = TensorBoardReporter()

    ### 実際には Agent + Env が合体したもの
    # agent = PPO(MultiEnv('Pendulum-v0', 10, reporter),
    agent = PPO(MultiEnv('Pendulum-v0', 4, reporter),
                reporter=reporter,
                normalize_state=True,
                normalize_reward=True,
                model_factory=MLP.factory(),
                curiosity_factory=ICM.factory(MlpICMModel.factory(), policy_weight=1, reward_scale=0.01, weight=0.2,
                                              intrinsic_reward_integration=0.01, reporter=reporter),
                # curiosity_factory=NoCuriosity.factory(),
                reward=GeneralizedRewardEstimation(gamma=0.95, lam=0.15),
                advantage=GeneralizedAdvantageEstimation(gamma=0.95, lam=0.15),
                learning_rate=4e-4,
                clip_range=0.3,
                v_clip_range=0.5,
                c_entropy=1e-2,
                c_value=0.5,
                n_mini_batches=32,
                n_optimization_epochs=10,
                clip_grad_norm=0.5)
    agent.to(device, torch.float32, np.float32)

    # agent.learn(epochs=30, n_steps=200)
    agent.learn(epochs=100, n_steps=200)
    # agent.eval(n_steps=600, render=True)
    for _ in range(10):
        states, actions, rewards, dones = agent.eval(n_steps=600, render=False)
        for i in range(4):
            s = states[i]
            a = actions[i]
            r = rewards[i]
            df = pd.DataFrame(index=range(states[0].shape[0]), columns=["action", "cos", "sin", "omega", "reward"])
            df.iloc[1:, 0]   = a.reshape((-1))
            df.iloc[0:, 1:4] = s.reshape((-1, 3))
            df.iloc[1:, 4]   = r.reshape((-1))
            df.to_csv(f"aaa_{i}.csv")
