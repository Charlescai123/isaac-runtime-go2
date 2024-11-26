from __future__ import annotations

import time

import torch
import copy
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
from typing import Dict, Union


from rsl_rl.algorithms.dpg import AbstractDPG
from rsl_rl.env import VecEnv
from rsl_rl.modules.network import Network
from rsl_rl.storage.storage import Dataset
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.storage.rollout_storage_old import RolloutStorageOld


class DDPG(AbstractDPG):
    """Deep Deterministic Policy Gradients algorithm.

    This is an implementation of the DDPG algorithm by Lillicrap et. al. for vectorized environments.

    Paper: https://arxiv.org/pdf/1509.02971.pdf
    """
    torch.autograd.set_detect_anomaly(True)
    def __init__(
            self,
            env: VecEnv,
            actor_lr: float = 1e-4,
            # actor_lr: float = 3e-4,
            critic_lr: float = 1e-3,
            **kwargs,
    ) -> None:
        print(f"env is: {env}")
        print(f"**kargs: {kwargs}")

        super().__init__(env, **kwargs)
        self._critic_input_size = 18
        print(f"self._actor_input_size: {self._actor_input_size}")
        print(f"self._action_size: {self._action_size}")
        print(f"self._critic_input_size: {self._critic_input_size}")

        print(f"self._actor_network_kwargs: {self._actor_network_kwargs}")
        print(f"self._critic_network_kwargs: {self._critic_network_kwargs}")
        # time.sleep(123)
        self.actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.critic = Network(self._critic_input_size, 1, **self._critic_network_kwargs)

        self.target_actor = Network(self._actor_input_size, self._action_size, **self._actor_network_kwargs)
        self.target_critic = Network(self._critic_input_size, 1, **self._critic_network_kwargs)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # for name, param in self.actor.named_parameters():
        #     print(f"{name}: {param.shape}")
        # time.sleep(123)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.trans_list = []
        self.transition = RolloutStorage.Transition()
        init_noise_std = 0.1
        self.lam = 0.95
        self.std = nn.Parameter(init_noise_std * torch.ones(self._action_size, device=self.device))

        self.transition = RolloutStorage.Transition()

        self._register_serializable(
            "actor", "critic", "target_actor", "target_critic", "actor_optimizer", "critic_optimizer"
        )

        self.to(self.device)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
        #                               action_shape, self.device)
        # self.storage = RolloutStorageOld(num_envs, device=self.device)
        pass

    def eval_mode(self) -> DDPG:
        super().eval_mode()

        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

        return self

    def act(self, obs, critic_obs):

        # print(f"obs: {obs.shape}")
        # print(f"critic_obs: {critic_obs.shape}")
        #
        # # if self.actor_critic.is_recurrent:
        # #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # # Compute the actions and values
        # print(f"self.nn_act(obs).detach().detach(): {self.evaluate(critic_obs).detach().shape}")
        # self.transition.actions = self.nn_act(obs).detach()
        # self.transition.values = self.evaluate(critic_obs).detach()
        # self.transition.actions_log_prob = self.get_actions_log_prob(self.transition.actions).detach()
        # self.transition.action_mean = self.action_mean.detach()
        # self.transition.action_sigma = self.action_std.detach()
        # # need to record obs and critic_obs before env.step()
        # self.transition.observations = obs
        # self.transition.critic_observations = critic_obs
        return self.nn_act(obs).detach()

    def to(self, device: str) -> DDPG:
        """Transfers agent parameters to device."""
        super().to(device)

        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)

        return self

    def train_mode(self) -> DDPG:
        super().train_mode()

        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

        return self

    def update(self, dataset: Dataset) -> [float, float]:
        # print(f"self.storage: {self.storage}")
        super().update(dataset)

        # if not self.initialized:
        #     return {}

        # KL
        # if self.desired_kl is not None and self.schedule == 'adaptive':
        #     with torch.inference_mode():
        #         kl = torch.sum(
        #             torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
        #                     torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
        #                     2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
        #         kl_mean = torch.mean(kl)
        #
        #         if kl_mean > self.desired_kl * 2.0:
        #             self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        #         elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
        #             self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        #
        #         for param_group in self.optimizer.param_groups:
        #             param_group['lr'] = self.learning_rate

        total_actor_loss = torch.zeros(self._batch_count)
        total_critic_loss = torch.zeros(self._batch_count)
        # print(f"batch_size: {self._batch_size}....................................................")
        for idx, batch in enumerate(self.storage.batch_generator(self._batch_size, self._batch_count)):
            # print(f"idx: {idx}")
            # print(f"batch: {batch}")
            actor_obs = batch["actor_observations"]
            critic_obs = batch["critic_observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            actor_next_obs = batch["next_actor_observations"]
            critic_next_obs = batch["next_critic_observations"]
            dones = batch["dones"].int()

            target_actor_prediction = self._process_actions(self.target_actor.forward(actor_next_obs))
            target_critic_prediction = self.target_critic.forward(
                self._critic_input(critic_next_obs, target_actor_prediction)
            )
            # print(f"target_actor:{target_actor_prediction}")
            target = rewards + self._discount_factor * (1 - dones) * target_critic_prediction
            print(f"actions: {actions}")
            print(f"critic_obs: {critic_obs}")
            prediction = self.critic.forward(self._critic_input(critic_obs, actions))
            critic_loss = (prediction - target).pow(2).mean()

            # Optimize Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            evaluation = self.critic.forward(
                self._critic_input(critic_obs, self._process_actions(self.actor.forward(actor_obs)))
            )
            actor_loss = -evaluation.mean()

            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self._update_target(self.actor, self.target_actor)
            self._update_target(self.critic, self.target_critic)

            total_actor_loss[idx] = actor_loss.item()
            total_critic_loss[idx] = critic_loss.item()

        # stats = {"actor": total_actor_loss.mean().item(), "critic": total_critic_loss.mean().item()}
        return total_critic_loss.mean().item(), total_actor_loss.mean().item()

    def process_env_step2(self, prev_obs, obs, actions, rewards, dones, infos):
        res = {
            'actor_observations': prev_obs.clone(),
            'critic_observations': prev_obs.clone(),
            'actions': actions.clone(),
            'rewards': rewards.clone(),
            'next_actor_observations': obs.clone(),
            'next_critic_observations': obs.clone(),
            'dones': dones.clone(),
            'timeouts': infos['time_outs'].clone()
        }

        return res

    def process_env_step(self, rewards, dones, infos):

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # print(f"self.transition: {self.transition.rewards.shape}")
        # print(f"self.transition.values: {self.transition.values.shape}")
        # print(f"infos['time_outs']: {infos['time_outs'].shape}")
        # print(f"infos['time_outs'].unsqueeze(1): {infos['time_outs'].unsqueeze(1).to(self.device).shape}")
        # print(f"rewards: {rewards.shape}")
        # print(f"dones: {dones.shape}")
        # print(f"infos: {infos}")

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            # print(f"un: {torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1).shape}")
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.reset(dones)

        # self.trans_list.append(copy.deepcopy(self.transition))
        # self.transition.clear()
        # self.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        # print(f"self.std: {self.std}")
        self.distribution = Normal(mean, mean * 0. + self.std)

    def nn_act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        # print(f"observations: {observations}")
        # print(f"actions_mean: {actions_mean}")
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        value = torch.unsqueeze(value, 1)
        # print(f"value is: {value.shape}")
        return value

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    def reset(self, dones=None):
        pass
