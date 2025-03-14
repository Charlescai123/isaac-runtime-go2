import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

from rsl_rl.algorithms.ddpg import DDPG
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


class OffPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        alg_class = eval(self.cfg["algorithm_class_name"])  # DDPG
        self.alg: DDPG = alg_class(env, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

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
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        ep_reward_sum = 0  # Episodic sum of reward
        ep_reward_avg = 0  # Episodic avg of reward
        failed_times = 0
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        total_iter = self.current_learning_iteration + num_learning_iterations
        for it in tqdm(range(self.current_learning_iteration, total_iter)):
            start = time.time()

            # with torch.inference_mode():
            # Rollout
            for i in range(self.num_steps_per_env):

                drl_actions = self.alg.act(obs, critic_obs)
                prev_obs = self.env.get_observations()
                obs, privileged_obs, actions, rewards, dones, infos = self.env.step(drl_actions)
                critic_obs = privileged_obs if privileged_obs is not None else obs
                prev_obs, obs, critic_obs, actions, rewards, dones = prev_obs.to(self.device), obs.to(
                    self.device), critic_obs.to(self.device), actions.to(self.device), rewards.to(
                    self.device), dones.to(self.device)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos and infos['episode'] != {}:
                        ep_infos.append(infos['episode'])
                    if 'fails' in infos:
                        failed_times = infos['fails']

                    cur_reward_sum += rewards  # sum reward
                    ep_reward_sum += rewards  # episodic reward
                    cur_episode_length += 1

                    # Reset current sum when dones
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                    # import pdb
                    # pdb.set_trace()

                    if torch.any(new_ids):
                        ep_step = ep_infos[0]["ep_step"][new_ids].cpu().numpy().tolist()
                        ep_reward_avg = ep_reward_sum / ep_step
                        ep_reward_sum = 0


                stop = time.time()
                collection_time = stop - start

                # Save dataset to replay buffer
                dataset = self.alg.to_transition(prev_obs=prev_obs, obs=obs, actions=actions, rewards=rewards,
                                                 dones=dones, infos=infos)
                # DDPG Update and return loss
                stats = self.alg.update([dataset])
                mean_value_loss, mean_surrogate_loss = stats["critic"], stats["actor"]

                # Learning time
                start = stop
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            # ep_reward_sum = 0

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.total_timesteps += self.num_steps_per_env * self.env.num_envs
        self.total_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/actor_learning_rate', self.alg.actor_optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('Loss/critic_learning_rate', self.alg.critic_optimizer.param_groups[0]['lr'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        self.writer.add_scalar('Perf/failed_times', locs['failed_times'], locs['it'])
        self.writer.add_scalar('Train/sum_reward_per_episode', locs['ep_reward_sum'], locs['it'])
        self.writer.add_scalar('Train/sum_reward_per_episode/time', locs['ep_reward_sum'], self.total_time)
        self.writer.add_scalar('Train/avg_reward_per_episode', locs['ep_reward_avg'], locs['it'])
        self.writer.add_scalar('Train/avg_reward_per_episode/time', locs['ep_reward_avg'], self.total_time)

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
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
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
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
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
        torch.save({
            'actor_state_dict': self.alg.actor.state_dict(),
            'critic_state_dict': self.alg.critic.state_dict(),
            'actor_optimizer_state_dict': self.alg.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.alg.critic_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

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

    def get_inference_policy(self, device=None):
        self.alg.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.to(device)
        return self.alg.act_inference
