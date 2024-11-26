"""
This script is adapted from clearRL
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from dataclasses import dataclass
import os


@dataclass
class SACParams:
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_steps: int = 1000
    """max timesteps of the episodes"""  # here we break the infinite continuous horizon into different episodes
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
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
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True) if len(log_prob.shape) > 1 else log_prob
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SACAgent:

    def __init__(self, params: SACParams, env, device):
        self.params = params
        self.env = env

        self.env.single_action_space = self.env.action_space
        self.env.single_observation_space = self.env.observation_space
        self.actor = Actor(env).to(device)
        self.qf1 = SoftQNetwork(env).to(device)
        self.qf2 = SoftQNetwork(env).to(device)
        self.qf1_target = SoftQNetwork(env).to(device)
        self.qf2_target = SoftQNetwork(env).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.params.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.params.policy_lr)

        self.replay_buffer = ReplayBuffer(self.params.buffer_size,
                                          env.observation_space,
                                          env.action_space, device,
                                          handle_timeout_termination=False)
        self.device = device

    def train_step(self, global_step):
        actor_loss = 0
        data = self.replay_buffer.sample(self.params.batch_size)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.params.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.params.gamma * (
                min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        loss_dict = {"q_values": qf1_a_values.mean().item(),
                     "critic_loss": qf_loss.item(),
                     "actor_loss": actor_loss}  #

        if global_step % self.params.policy_frequency == 0:
            for _ in range(self.params.policy_frequency):
                # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.actor.get_action(data.observations)
                qf1_pi = self.qf1(data.observations, pi)
                qf2_pi = self.qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.params.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            loss_dict["actor_loss"] = actor_loss.item()

            # update the target network
        if global_step % self.params.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)

        return loss_dict

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/weights.cleanrl_model"
        torch.save((self.actor.state_dict(), self.qf1.state_dict(), self.qf2.state_dict()), model_path)

    def load_model(self, model_path):
        weights = torch.load(model_path)
        self.actor.load_state_dict(weights[0])
        self.qf1.load_state_dict(weights[1])
        self.qf2.load_state_dict(weights[2])
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

    def get_actions(self, obs, mode='train'):
        # get actions to interaction with the env
        with torch.no_grad():
            if mode == "train":
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.cpu().numpy()
            else:
                mean, _ = self.actor(torch.Tensor(obs).to(self.device))
                y_t = torch.tanh(mean)
                actions = y_t * self.actor.action_scale + self.actor.action_bias
                actions = actions.cpu().numpy()
            return actions
