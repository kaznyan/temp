from multiprocessing import Process, Pipe
from typing import Tuple, Any, List, Dict, Union

import numpy as np
from gym import Env, make

from envs.utils import tile_images
from reporters import Reporter, NoReporter


class SubProcessEnv(Process):
    """
    Process that runs the environment. It is controlled by ``MultiEnv`` using ``Pipe``'s

    *Note*: If the environment steps into terminal state (``done == True``) it immediately resets the envoronment
            and continues as if nothing happened. It is necessary, because multiple environments run at the same time
            can end in different time steps and ``MultiEnv`` must continue on them all
    """
    DONE_IDX = 2

    def __init__(self, env_id, master, slave):
        super().__init__(daemon=True)
        self.master = master
        self.env_id = env_id
        self.pipe = slave

    def start(self) -> None:
        super().start()
        self.pipe.close()

    def run(self):
        self.master.close()
        env = make(self.env_id)
        steps = 0
        collected_reward = 0
        while True:
            command, args = self.pipe.recv()
            if command == 'getattr':
                self.pipe.send(getattr(env, args))
            elif command == 'seed':
                self.pipe.send(env.seed(args))
            elif command == 'reset':
                steps = 0
                collected_reward = 0
                self.pipe.send(env.reset())
            elif command == 'render':
                self.pipe.send(env.render(args))
            elif command == 'step':
                state, reward, done, aux = env.step(args)
                steps += 1
                collected_reward += reward
                if done:
                    state = env.reset()
                self.pipe.send((state, reward, done, aux, collected_reward))
                if done:
                    steps = 0
                    collected_reward = 0
            elif command == 'close':
                env.close()
                break


class MultiEnv(Env):
    """
    Runs multiple environments simultaneously. Most of the methods of this envoronment returns ``np.ndarray`` with
    aggregated responses from environments instead of single values.

    *Note*: Since this environment runs multiple ones underneath it never finishes. If one of the environments returns
            ``done`` signal. It is remembered, but the environment immediately resets and continues starting new episode
    """

    def __init__(self, env_id, n_envs: int, reporter: Reporter = NoReporter()):
        """

        :param env_id: name of the environment that ``gym.make`` will be called with
        :param n_envs: number of environments to run simultaneously
        :param reporter reporter to be used to report rewards
        """
        self._closed = False
        self.env_id = env_id
        self.n_envs: int = n_envs
        self.reporter = reporter
        self.processes = [SubProcessEnv(env_id, *Pipe()) for _ in range(self.n_envs)]
        self._start()
        self.observation_space = self._get_property(self.processes[0], 'observation_space')
        self.action_space = self._get_property(self.processes[0], 'action_space')
        self.dtype = None

    def _start(self):
        for process in self.processes:
            process.start()

    def _send_command(self, name: str, args=None, await_response: bool = True) -> List[Any]:
        for process, arg in zip(self.processes, args if args is not None else [None] * len(self.processes)):
            process.master.send((name, arg))
        return [self._rcv(process) for process in self.processes] if await_response else []

    def _rcv(self, process):
        res = process.master.recv()
        return res

    def _get_property(self, process, name):
        process.master.send(('getattr', name))
        return self._rcv(process)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Make one step in all environments

        :param action: ``array`` of actions for each agent of shape N*(action space shape) where N is number of
               environments. First dimension length must be equal to number of environments
        :return: tuple of:

                 * ``state``  array of shape N*(state space shape)
                 * ``reward`` array of shape (N,)
                 * ``done`` array of shape (N,)
                 * ``aux`` list of auxiliary information returned by the agents
        """

        if len(action) != self.n_envs:
            raise ValueError('Not enough actions supplied')
        state, reward, done, aux, collected_rewards = zip(*self._send_command('step', action))
        self._report_steps(done, collected_rewards)
        return np.array(state, dtype=self.dtype), np.array(reward, dtype=self.dtype), \
               np.array(done, dtype=self.dtype), aux

    def reset(self):
        return np.array(self._send_command('reset'), dtype=self.dtype)

    def render(self, mode='human'):
        imgs = self._send_command('render', ['rgb_array'] * self.n_envs)
        if any(img is None for img in imgs):
            return None
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow(self.env_id, bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def close(self):
        if self._closed:
            return
        self._send_command('close', await_response=False)
        for process in self.processes:
            process.join()
        self._closed = True

    def seed(self, seed=None):
        return np.array(self._send_command('seed', [seed] * self.n_envs), dtype=self.dtype)

    def astype(self, dtype: Union[object, str]):
        self.dtype = dtype

    def _report_steps(self, dones, collected_rewards):
        for done, reward in zip(dones, collected_rewards):
            if done:
                self.reporter.scalar('env/reward', reward)
