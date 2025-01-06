import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import tensorflow as tf

from rsl_rl.tf_version.ddpg_tf import DDPGTF
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.tf_version.actor_critic_tf import AbstractActorCritic

class OffPolicyRunnerTF:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        # actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
        #                                                num_critic_obs,
        #                                                self.env.num_actions,
        #                                                **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # DDPGTF
        # self.alg: DDPG = alg_class(env, device=self.device, **self.alg_cfg)
        self.alg: DDPGTF = alg_class(env, device=self.device)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.total_timesteps = 0
        self.total_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        # obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        obs = tf.convert_to_tensor(obs.cpu().numpy())
        critic_obs = tf.convert_to_tensor(critic_obs.cpu().numpy())
        # self.alg.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        total_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, total_iter):

            transitions_list = []

            start = time.time()
            # Rollout
            # with torch.inference_mode():

            for i in range(self.num_steps_per_env):
                actions = self.alg.act(obs, critic_obs)
                actions = torch.from_numpy(actions.numpy())
                prev_obs = self.env.get_observations()
                obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                # print(f"prev_obs: {prev_obs}")
                # print(f"next_obs: {obs}")
                # print(f"infos: {infos}")
                critic_obs = privileged_obs if privileged_obs is not None else obs

                # From pytorch to tf
                obs = tf.convert_to_tensor(obs.cpu().numpy())
                critic_obs = tf.convert_to_tensor(critic_obs.cpu().numpy())
                rewards = tf.convert_to_tensor(rewards.cpu().numpy())
                dones = tf.convert_to_tensor(dones.cpu().numpy())
                prev_obs = tf.convert_to_tensor(prev_obs.cpu().numpy())
                # prev_obs, obs, critic_obs, rewards, dones = prev_obs.to(self.device), obs.to(
                #     self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                # self.alg.process_env_step(rewards, dones, infos)
                # print(f"infos: {infos}")
                # print(f"infos: {type(infos)}")
                # for info in list(infos.keys()):
                #     print(f"info: {info}")
                    # for k in infos[info].keys():
                    #     info[k] = tf.convert_to_tensor(infos[info][k].cpu().numpy())

                for k in infos['episode'].keys():
                    infos['episode'][k] = tf.convert_to_tensor(infos['episode'][k].cpu().numpy())

                infos['time_outs'] = tf.convert_to_tensor(infos['time_outs'].cpu().numpy())

                transitions_list.append(
                    self.alg.process_env_step2(prev_obs=prev_obs, obs=obs, actions=actions, rewards=rewards,
                                               dones=dones, infos=infos))

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += torch.from_numpy(rewards.numpy()).to(self.device)
                    cur_episode_length += 1
                    # new_ids = (dones > 0).nonzero(as_tuple=False)
                    new_ids = tf.where(dones)
                    new_ids = torch.from_numpy(new_ids.numpy()).to(self.device)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                mean_value_loss, mean_surrogate_loss = self.alg.update([transitions_list[-1]])
                mean_value_loss = torch.tensor(mean_value_loss.numpy())
                mean_surrogate_loss = torch.tensor(mean_surrogate_loss.numpy())

                # Learning step
                start = stop
                # self.alg.compute_returns(critic_obs)

            # mean_value_loss, mean_surrogate_loss = self.alg.update(transitions_list)
            # self.alg.trans_list.clear()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.total_timesteps += self.num_steps_per_env * self.env.num_envs
        self.total_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            print(f"ep_infos: {locs['ep_infos']}")
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], tf.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    else:
                        ep_info[key] = torch.tensor(ep_info[key].numpy()).to(self.device)
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = tf.reduce_mean(self.alg.std)
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/actor_learning_rate', self.alg.actor_optimizer.learning_rate.numpy(), locs['it'])
        self.writer.add_scalar('Loss/critic_learning_rate', self.alg.critic_optimizer.learning_rate.numpy(), locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.numpy(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.total_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']),
                                   self.total_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.numpy():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.numpy():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.total_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.total_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.total_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        self.alg.actor.save_weights(os.path.join(path, "actor"))
        self.alg.critic.save_weights(os.path.join(path, "critic"))
        self.alg.target_actor.save_weights(os.path.join(path, "actor_target"))
        self.alg.target_critic.save_weights(os.path.join(path, "critic_target"))
        # torch.save({
        #     'actor_state_dict': self.alg.actor.state_dict(),
        #     'critic_state_dict': self.alg.critic.state_dict(),
        #     'actor_optimizer_state_dict': self.alg.actor_optimizer.state_dict(),
        #     'critic_optimizer_state_dict': self.alg.critic_optimizer.state_dict(),
        #     'iter': self.current_learning_iteration,
        #     'infos': infos,
        # }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        print(f"path: {path}")
        print(f"loaded_dict: {loaded_dict.keys()}")
        loaded_dict = torch.load(path)

        for k, v in loaded_dict['actor_state_dict'].items():
            print(f"k: {k}, v: {v}")

        self.alg.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.alg.critic.load_state_dict(loaded_dict['critic_state_dict'])
        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict['actor_optimizer_state_dict'])
            self.alg.critic_optimizer.load_state_dict(loaded_dict['critic_optimizer_state_dict'])

        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']
        # self.alg.actor.load_state_dict(loaded_dict['actor_state_dict'])
        # self.alg.critic.load_state_dict(loaded_dict['critic_state_dict'])
        # if load_optimizer:
        #     self.alg.actor_optimizer.load_state_dict(loaded_dict['actor_optimizer_state_dict'])
        #     self.alg.critic_optimizer.load_state_dict(loaded_dict['critic_optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        # return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.to(device)
        return self.alg.act_inference
