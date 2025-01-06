# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import time

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic_tf import ActorCriticTF
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.rollout_storage_tf import RolloutStorageTF


class PPOTF:
    actor_critic: ActorCriticTF

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic

        # for name, param in self.actor_critic.named_parameters():
        #     print(f"{name}: {param.shape}")
        # time.sleep(123)

        # self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        # self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.std_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageTF(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape,
                                        action_shape, self.device)

    # def test_mode(self):
    #     self.actor_critic.eval()
    #
    # def train_mode(self):
    #     self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            pass
            # self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # print(f"obs: {obs.shape}")
        # print(f"critic_obs: {critic_obs.shape}")
        # import time
        # time.sleep(123)
        # print(f"self.actor_critic.act(obs).detach(): {self.actor_critic.act(obs).detach().shape}")
        self.transition.actions = self.actor_critic.act(obs).numpy()
        self.transition.values = self.actor_critic.evaluate(critic_obs).numpy()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).numpy()
        # print(f"self.actor_critic.action_mean: {self.actor_critic.action_mean}")
        # print(f"self.actor_critic.action_mean: {type(self.actor_critic.action_mean)}")
        self.transition.action_mean = self.actor_critic.action_mean.numpy()
        self.transition.action_sigma = self.actor_critic.action_std.numpy()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.numpy()
        self.transition.dones = dones.numpy()

        # print(f"self.transition: {self.transition.rewards.shape}")
        # print(f"self.transition.values: {self.transition.values.shape}")
        # print(f"infos['time_outs']: {infos['time_outs'].shape}")
        # print(f"infos['time_outs'].unsqueeze(1): {infos['time_outs'].unsqueeze(1).to(self.device).shape}")
        # print(f"rewards: {rewards.shape}")
        # print(f"dones: {dones.shape}")
        # print(f"infos: {infos}")

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            print(f"infos['time_outs']: {infos['time_outs']}")
            print(f"self.transition.rewards: {type(infos['time_outs'].cpu().numpy())}")

            # timeout_list = infos['time_outs'].cpu().numpy()
            # tf_boolean_tensor = tf.cast()

            self.transition.rewards += (self.gamma * tf.squeeze(
                self.transition.values * tf.expand_dims(
                    tf.convert_to_tensor(infos['time_outs'].cpu().numpy(), dtype=tf.float32), axis=1),
                axis=1))

        # import time
        # time.sleep(123)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).numpy()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            with tf.GradientTape(persistent=True) as tape:
                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch,
                                                         hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl is not None and self.schedule == 'adaptive':
                    kl = tf.reduce_sum(
                        tf.math.log(sigma_batch / (old_sigma_batch + 1.e-5)) +
                        (tf.square(old_sigma_batch) + tf.square(old_mu_batch - mu_batch)) /
                        (2.0 * tf.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = tf.reduce_mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif self.desired_kl / 2.0 > kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    self.actor_optimizer.learning_rate.assign(tf.Variable(self.learning_rate, dtype=tf.float32))
                    self.critic_optimizer.learning_rate.assign(tf.Variable(self.learning_rate, dtype=tf.float32))
                    # for param_group in self.optimizer.param_groups:
                    #     param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = tf.exp(actions_log_prob_batch - tf.squeeze(old_actions_log_prob_batch))
                surrogate = -tf.squeeze(advantages_batch) * ratio
                surrogate_clipped = -tf.squeeze(advantages_batch) * tf.clip_by_value(ratio, 1.0 - self.clip_param,
                                                                                     1.0 + self.clip_param)
                surrogate_loss = tf.reduce_mean(tf.maximum(surrogate, surrogate_clipped))

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + tf.clip_by_value((value_batch - target_values_batch),
                                                                           -self.clip_param, self.clip_param)
                    value_losses = tf.square(value_batch - returns_batch)
                    value_losses_clipped = tf.square(value_clipped - returns_batch)
                    value_loss = tf.reduce_mean(tf.maximum(value_losses, value_losses_clipped))
                else:
                    value_loss = tf.reduce_mean(tf.square(returns_batch - value_batch))

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * tf.reduce_mean(
                    entropy_batch)

            # Gradient step to actor and critic
            gradients = tape.gradient(loss, self.actor_critic.actor.trainable_variables)
            clipped_gradients = clip_gradients(gradients, self.max_grad_norm)
            self.actor_optimizer.apply_gradients(zip(clipped_gradients, self.actor_critic.actor.trainable_variables))
            gradients = tape.gradient(loss, self.actor_critic.critic.trainable_variables)
            clipped_gradients = clip_gradients(gradients, self.max_grad_norm)
            self.critic_optimizer.apply_gradients(zip(clipped_gradients, self.actor_critic.critic.trainable_variables))
            # print(f"self.actor_critic.actor.trainable_variables: {self.actor_critic.actor.trainable_variables}")
            # print(f"self.actor_critic.actor.trainable_variables: {type(self.actor_critic.actor.trainable_variables)}")
            # print(f"self.actor_critic.std: {self.actor_critic.std}")
            # print(f"self.actor_critic.std: {type(self.actor_critic.std)}")

            # Gradient step to noise std
            gradients = tape.gradient(loss, [self.actor_critic.std])
            clipped_gradients = clip_gradients(gradients, self.max_grad_norm)
            self.std_optimizer.apply_gradients(zip(clipped_gradients, [self.actor_critic.std]))

            mean_value_loss += value_loss.numpy()
            mean_surrogate_loss += surrogate_loss.numpy()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss


def clip_gradients(gradients, max_grad_norm):
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
    return clipped_gradients
