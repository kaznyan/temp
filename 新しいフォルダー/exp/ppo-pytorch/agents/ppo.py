from itertools import chain

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from curiosity import CuriosityFactory
from envs import Runner, MultiEnv, Converter, RandomRunner
from models import ModelFactory
from normalizers import StandardNormalizer, Normalizer, NoNormalizer
from reporters import Reporter, NoReporter
from rewards import Reward, Advantage

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return np.asarray([self.action_space.sample() for _ in state])

class PPO(object):
    def __init__(self, env, model_factory, curiosity_factory, reward, advantage, learning_rate,
                 clip_range, v_clip_range, c_entropy, c_value, n_mini_batches, n_optimization_epochs,
                 clip_grad_norm, normalize_state, normalize_reward, reporter=NoReporter()):
        """
        :param env: environment to train on
        :param model_factory: factory to construct the model used as the brain of the agent
        :param curiosity_factory: factory to construct curiosity object
        :param reward: reward function to use for discounted reward calculation
        :param advantage: advantage function to use for advantage calculation
        :param learning_rate: learning rate
        :param clip_range: clip range for surrogate function and value clipping
        :param v_clip_range: clip range for value function clipping
        :param c_entropy: entropy coefficient constant
        :param c_value: value coefficient constant
        :param n_mini_batches: number of mini batches to devide experience into for optimization
        :param n_optimization_epochs number of optimization epochs on same experience. This value is called ``K`` in paper
        :param clip_grad_norm: value used to clip gradient by norm
        :param normalize_state whether to normalize the observations or not
        :param normalize_reward whether to normalize rewards or not
        :param reporter: reporter to be used for reporting learning statistics
        """
        self.env = env
        self.state_converter  = Converter.for_space(self.env.observation_space)
        self.action_converter = Converter.for_space(self.env.action_space)

        self.model = model_factory.create(self.state_converter, self.action_converter)
        self.curiosity = curiosity_factory.create(self.state_converter, self.action_converter)

        self.reward = reward
        self.advantage = advantage

        self.n_mini_batches = n_mini_batches
        self.n_optimization_epochs = n_optimization_epochs

        self.clip_grad_norm = clip_grad_norm

        self.optimizer = Adam(chain(self.model.parameters(), self.curiosity.parameters()), learning_rate)
        self.loss = PPOLoss(clip_range, v_clip_range, c_entropy, c_value, reporter)

        self.state_normalizer = self.state_converter.state_normalizer() if normalize_state else NoNormalizer()
        self.normalize_state = normalize_state
        self.reward_normalizer = StandardNormalizer() if normalize_reward else NoNormalizer()

        self.reporter = reporter

        self.device = None
        self.dtype = None
        self.numpy_dtype = None

    def act(self, state):
        """
        Acts in the environment. Returns the action for the given state
        Note: ``N`` in the dimensions stands for number of parallel environments being explored
        :param state: state of shape N * (state space shape) that we want to know the action for
        :return: the action which is array of shape N * (action space shape)
        """
        state = self.state_normalizer.transform(state[:, None, :])
        reshaped_states = self.state_converter.reshape_as_input(state, self.model.recurrent)
        logits = self.model.policy_logits(torch.tensor(reshaped_states, device=self.device))
        return self.action_converter.action(logits).cpu().detach().numpy()

    def learn(self, epochs, n_steps, initialization_steps=1000, render=False):
        """
        Trains the agent for ``epochs`` number of times by running simulation on the environment for ``n_steps``
        :param epochs: number of epochs of training
        :param n_steps: number of steps made in the environment each epoch
        :param initialization_steps: number of steps made on the environment to gather the states then used for
               initialization of the state normalizer
        :param render: whether to render the environment during learning
        """
        if initialization_steps and self.normalize_state:
            print(">>>Running random agent ...")
            s, _, _, _ = RandomRunner(self.env).run(initialization_steps)
            self.state_normalizer.partial_fit(s)

        for epoch in range(epochs):
            states, actions, rewards, dones = Runner(self.env, self).run(n_steps, render)
            print(np.argmax(dones, axis=1).mean(), dones.sum())
            states = self.state_normalizer.partial_fit_transform(states)
            rewards = self.curiosity.reward(rewards, states, actions)
            rewards = self.reward_normalizer.partial_fit_transform(rewards)
            self._train(states, actions, rewards, dones)
            print(f'Epoch: {epoch} done')

    def eval(self, n_steps, render=False):
        return Runner(self.env, self).run(n_steps, render)

    def to(self, device, dtype, numpy_dtype):
        """
        Transfers the agent's model to device
        :param device: device to transfer agent to
        :param dtype: dtype to which cast the model parameters
        :param numpy_dtype: dtype to use for the environment. *Must* be the same as ``dtype`` parameter
        """
        self.device = device
        self.dtype = dtype
        self.numpy_dtype = numpy_dtype
        self.model.to(device, dtype)
        self.curiosity.to(device, dtype)
        self.env.astype(numpy_dtype)

    def _train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        policy_old, values_old = self.model(self._to_tensor(self.state_converter.reshape_as_input(states, self.model.recurrent)))
        policy_old = policy_old.detach().view(*states.shape[:2], -1)
        values_old = values_old.detach().view(*states.shape[:2])
        values_old_numpy = values_old.cpu().detach().numpy()
        discounted_rewards = self.reward.discounted(rewards, values_old_numpy, dones)
        advantages = self.advantage.discounted(rewards, values_old_numpy, dones)
        dataset = self.model.dataset(policy_old[:, :-1], values_old[:, :-1], states[:, :-1], states[:, 1:], actions,
                                     discounted_rewards, advantages)
        loader = DataLoader(dataset, batch_size=len(dataset) // self.n_mini_batches, shuffle=True)
        with torch.autograd.detect_anomaly():
            for _ in range(self.n_optimization_epochs):
                for tuple_of_batches in loader:
                    (batch_policy_old, batch_values_old, batch_states, batch_next_states,
                     batch_actions, batch_rewards, batch_advantages) = self._tensors_to_device(*tuple_of_batches)
                    batch_policy, batch_values = self.model(batch_states)
                    batch_values = batch_values.squeeze()
                    distribution_old = self.action_converter.distribution(batch_policy_old)
                    distribution = self.action_converter.distribution(batch_policy)
                    loss: Tensor = self.loss(distribution_old, batch_values_old, distribution, batch_values,
                                             batch_actions, batch_rewards, batch_advantages)
                    loss = self.curiosity.loss(loss, batch_states, batch_next_states, batch_actions)
                    # print('loss:', loss)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()

    def _tensors_to_device(self, *tensors):
        return [tensor.to(self.device, self.dtype) for tensor in tensors]

    def _to_tensor(self, array):
        return torch.tensor(array, device=self.device, dtype=self.dtype)

class PPOLoss(_Loss):
    r"""
    Calculates the PPO loss given by equation:
    .. math:: L_t^{CLIP+VF+S}(\theta) = \mathbb{E} \left [L_t^{CLIP}(\theta) - c_v * L_t^{VF}(\theta)
                                        + c_e S[\pi_\theta](s_t) \right ]
    where:
    .. math:: L_t^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left [\text{min}(r_t(\theta)\hat{A}_t,
                                  \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t )\right ]
    .. math:: r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t
    .. math:: \L_t^{VF}(\theta) = (V_\theta(s_t) - V_t^{targ})^2
    and :math:`S[\pi_\theta](s_t)` is an entropy
    """
    def __init__(self, clip_range: float, v_clip_range: float, c_entropy: float, c_value: float, reporter: Reporter):
        """
        :param clip_range: clip range for surrogate function clipping
        :param v_clip_range: clip range for value function clipping
        :param c_entropy: entropy coefficient constant
        :param c_value: value coefficient constant
        :param reporter: reporter to be used to report loss scalars
        """
        super().__init__()
        self.clip_range = clip_range
        self.v_clip_range = v_clip_range
        self.c_entropy = c_entropy
        self.c_value = c_value
        self.reporter = reporter

    def forward(self, distribution_old: Distribution, value_old: Tensor, distribution: Distribution,
                value: Tensor, action: Tensor, reward: Tensor, advantage: Tensor):
        # Value loss
        value_old_clipped = value_old + (value - value_old).clamp(-self.v_clip_range, self.v_clip_range)
        v_old_loss_clipped = (reward - value_old_clipped).pow(2)
        v_loss = (reward - value).pow(2)
        value_loss = torch.min(v_old_loss_clipped, v_loss).mean()

        # Policy loss
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        advantage.detach_()
        log_prob = distribution.log_prob(action)
        log_prob_old = distribution_old.log_prob(action)
        ratio = (log_prob - log_prob_old).exp().view(-1)

        surrogate = advantage * ratio
        surrogate_clipped = advantage * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)
        policy_loss = torch.min(surrogate, surrogate_clipped).mean()

        # Entropy
        entropy = distribution.entropy().mean()

        # Total loss
        losses = policy_loss + self.c_entropy * entropy - self.c_value * value_loss
        total_loss = -losses
        self.reporter.scalar('ppo_loss/policy', -policy_loss.item())
        self.reporter.scalar('ppo_loss/entropy', -entropy.item())
        self.reporter.scalar('ppo_loss/value_loss', value_loss.item())
        self.reporter.scalar('ppo_loss/total', total_loss)
        return total_loss
