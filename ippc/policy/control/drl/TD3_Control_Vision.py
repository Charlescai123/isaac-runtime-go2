"""
This script is adapted from clearRL
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py
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
class TD3Params:
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_steps: int = 1000
    """max timesteps of the episodes"""  # here we break the infinite continuous horizon into different episodes
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    policy_noise: float = 0.2
    """the scale of policy noise"""


class QNetwork(nn.Module):
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


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class TD3Agent:
    def __init__(self, params: TD3Params, env, device):
        self.params = params
        self.env = env
        # we are not using vector env yet.

        self.env.single_action_space = self.env.action_space
        self.env.single_observation_space = self.env.observation_space
        self.actor = Actor(env).to(device)
        self.qf1 = QNetwork(env).to(device)
        self.qf2 = QNetwork(env).to(device)
        self.qf1_target = QNetwork(env).to(device)
        self.qf2_target = QNetwork(env).to(device)
        self.target_actor = Actor(env).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.params.learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.params.learning_rate)

        self.replay_buffer = ReplayBuffer(self.params.buffer_size,
                                          env.observation_space,
                                          env.action_space, device,
                                          handle_timeout_termination=False)
        self.device = device

    def train_step(self, global_step):
        # ALGO LOGIC: training.
        actor_loss = 0
        data = self.replay_buffer.sample(self.params.batch_size)
        with torch.no_grad():
            clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.params.policy_noise).clamp(
                -self.params.noise_clip, self.params.noise_clip
            ) * self.target_actor.action_scale

            next_state_actions = (self.target_actor(data.next_observations) + clipped_noise).clamp(
                torch.tensor(self.env.single_action_space.low), torch.tensor(self.env.single_action_space.high)
            )

            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.params.gamma * (
                min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        loss_dict = {"q_values": qf1_a_values.mean().item(),
                     "critic_loss": qf_loss.item(),
                     "actor_loss": actor_loss}  #

        if global_step % self.params.policy_frequency == 0:
            actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.params.tau * param.data + (1 - self.params.tau) * target_param.data)

            loss_dict["actor_loss"] = actor_loss.item()

        return loss_dict

    def save_model(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        model_path = f"{model_dir}/weights.cleanrl_model"
        torch.save((self.actor.state_dict(), self.qf1.state_dict(), self.qf2.state_dict()), model_path)

    def load_model(self, model_path):
        weights = torch.load(model_path)
        self.actor.load_state_dict(weights[0])
        self.qf1.load_state_dict(weights[1])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())

    def get_actions(self, obs, mode='train'):
        with torch.no_grad():
            actions = self.actor(torch.Tensor(obs).to(self.device))
            # we only add exploration noise during training
            if mode == 'train':
                actions += torch.normal(0, self.actor.action_scale * self.params.exploration_noise)
            actions = actions.cpu().numpy().clip(self.env.single_action_space.low, self.env.single_action_space.high)
            return actions
