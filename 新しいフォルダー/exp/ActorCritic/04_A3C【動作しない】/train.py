import gym
import torch

from model import Model
from worker import Worker
from shared_adam import SharedAdam
import torch.multiprocessing as mp

from config import env_name, lr

"""
multiproccessingがうまくいかず動かない
解決策は後ほどにして仕組みだけ理解する
"""

def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    ### 共通となるモデルを定義　これを各ワーカーに参照渡し（？）する
    global_model = Model(num_inputs, num_actions)
    global_model.share_memory()
    global_optimizer = SharedAdam(global_model.parameters(), lr=lr)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    ### 各Worlerを定義　global_model, およびそれを学習させるための各機能を参照（？）渡し
    workers = [Worker(global_model, global_optimizer, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    ### ここから大体の仕事はWorkerに移る
    [w.start() for w in workers] ### Worker.start() は Worker.run() を実行するというmultiprocessの仕様
    res = []

    while True:
        ### res_queueには各Workerの結果が集積されていく
        r = res_queue.get()
        if r is not None:
            res.append(r)
            [ep, ep_r, loss] = r
        else:
            break
    [w.join() for w in workers]

if __name__=="__main__":
    main()
