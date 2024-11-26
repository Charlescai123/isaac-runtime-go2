"""
This script is adapted from clearRL
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import os


@dataclass
class PPOParams:
    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_env: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = torch.sum(log_prob)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        entropy = torch.sum(normal.entropy())
        return action, log_prob, entropy

    def get_log_prob_entropy(self, x, action):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(1)
        entropy = normal.entropy().sum(1)
        return log_prob, entropy


class OnPolicyBuffer:
    def __init__(self, params: PPOParams, observation_space_shape, action_space_shape, device):
        self.params = params
        self.observation_space_shape = observation_space_shape
        self.action_space_shape = action_space_shape

        self.online_buffer = None
        self.reset_online_buffer()
        self.device = device
        self.step_counter = 0

        self.obs = None
        self.actions = None
        self.logprobs = None
        self.rewards = None
        self.dones = None
        self.values = None

        self.reset_online_buffer()

    def reset_online_buffer(self):
        self.obs = torch.zeros((self.params.num_steps,) + self.observation_space_shape)
        self.actions = torch.zeros((self.params.num_steps,) + self.action_space_shape)
        self.logprobs = torch.zeros((self.params.num_steps, ))
        self.rewards = torch.zeros((self.params.num_steps, ))
        self.dones = torch.zeros((self.params.num_steps, ))
        self.values = torch.zeros((self.params.num_steps, ))

    def add(self, obs, actions, rewards, logprob, terminations, values):
        self.obs[self.step_counter] = torch.Tensor(obs)
        self.dones[self.step_counter] = torch.Tensor(np.array(terminations))
        self.rewards[self.step_counter] = torch.Tensor(rewards)
        self.actions[self.step_counter] = torch.Tensor(actions)
        self.values[self.step_counter] = torch.Tensor(values)
        self.logprobs[self.step_counter] = torch.Tensor(logprob)
        self.step_counter += 1

    def get_size(self):
        return self.step_counter


class PPOAgent:
    def __init__(self, params: PPOParams, env, device):
        self.params = params
        self.env = env
        self.device = device
        self.env.single_action_space = self.env.action_space
        self.env.single_observation_space = self.env.observation_space
        self.params.batch_size = int(self.params.num_env * self.params.num_steps)
        self.params.minibatch_size = int(self.params.batch_size // self.params.num_minibatches)

        self.actor = Actor(env).to(device)
        self.critic = QNetwork(env).to(device)

        self.q_optimizer = optim.Adam(list(self.critic.parameters()), lr=self.params.learning_rate, eps=1e-5)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.params.learning_rate, eps=1e-5)

        self.replay_buffer = (
            OnPolicyBuffer(params, self.env.single_observation_space.shape, self.env.single_action_space.shape, self.device))

    def train_step(self):
        loss_dict = {}
        # calculate the advantages:
        with torch.no_grad():
            next_value = self.critic(self.replay_buffer.obs[-1]).reshape(1, -1)
            advantages = torch.zeros_like(self.replay_buffer.rewards).to(self.device)
            lastgaelam = 0

            for t in reversed(range(self.params.num_steps)):
                if t == self.params.num_steps - 1:
                    nextnonterminal = 1.0 - self.replay_buffer.dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.replay_buffer.dones[t + 1]
                    nextvalues = self.replay_buffer.values[t + 1]
                delta = self.replay_buffer.rewards[t] + self.params.gamma * nextvalues * nextnonterminal \
                        - self.replay_buffer.values[t]
                advantages[t] = lastgaelam = \
                    delta + self.params.gamma * self.params.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + self.replay_buffer.values

        # flatten the batch
        b_obs = self.replay_buffer.obs.reshape((-1, ) + self.env.single_observation_space.shape)
        b_logprobs = self.replay_buffer.logprobs.reshape(-1)
        b_actions = self.replay_buffer.actions.reshape((-1,) + self.env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.replay_buffer.values.reshape(-1)

        b_inds = np.arange(self.params.batch_size)
        clipfracs = []

        for epoch in range(self.params.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.params.batch_size, self.params.minibatch_size):
                end = start + self.params.minibatch_size
                mb_inds = b_inds[start:end]
                newvalue = self.critic(b_obs[mb_inds])
                newlogprob, entropy = self.actor.get_log_prob_entropy(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    # approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.params.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.params.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.params.clip_coef, 1 + self.params.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.params.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.params.clip_coef,
                        self.params.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                self.q_optimizer.zero_grad()
                v_loss.backward()
                self.q_optimizer.step()

                actor_loss = pg_loss - self.params.ent_coef * entropy_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                loss_dict = {"q_values": newvalue.mean().item(),
                             "critic_loss": v_loss.item(),
                             "actor_loss": pg_loss.item()}

        return loss_dict

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/weights.cleanrl_model"
        torch.save((self.actor.state_dict(), self.critic), model_path)

    def load_model(self, model_path):
        weights = torch.load(model_path)
        self.actor.load_state_dict(weights[0])
        self.critic.load_state_dict(weights[1])

    def get_actions(self, obs, mode='train'):

        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
            action = action.cpu().numpy().reshape(-1)
            if mode == 'train':
                value = self.critic(torch.Tensor(obs).to(self.device))
                value = value.cpu().numpy().reshape(-1)
                return action, log_prob, value
            else:
                return action

