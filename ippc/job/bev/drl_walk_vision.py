import dataclasses
import os
import sys
import click
import random
import torch
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad_gym.env_builder import build_a1_ground_env
from quad_gym.gym_config import GymConfig
from utils.utils import load_config, save_params
from policy.policy_config import PolicyConfig
from job.job_config import JobConfig
from job.job_monitor.job_logger import Logger
from dataclasses_json import dataclass_json
from policy.policy_config import AllControlModels


@dataclass_json
@dataclasses.dataclass
class AllConfig:
    GymParams: GymConfig = GymConfig()
    PolicyParams: PolicyConfig = PolicyConfig()
    JobParams: JobConfig = JobConfig()


@click.command()
@click.argument('config')
@click.option('--gpu', is_flag=True)
@click.option('--generate', is_flag=True, help="Generate a config file instead.")
@click.option('-p', '--params', nargs=2, multiple=True)
def run(**kwargs):
    if kwargs['generate']:
        save_params(AllConfig(), kwargs['config'])
        print("Config file generated.")
        return

    if not kwargs['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params: AllConfig = load_config(kwargs['config'], overrides=kwargs['params'])

    random.seed(params.JobParams.seed)
    np.random.seed(params.JobParams.seed)
    torch.manual_seed(params.JobParams.seed)
    torch.backends.cudnn.deterministic = params.JobParams.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and params.JobParams.cuda else "cpu")

    env = build_a1_ground_env(params.GymParams)

    control_model_name = params.PolicyParams.ControlModuleParams.control_model
    control_model_params = params.PolicyParams.ControlModuleParams.control_model_params

    agent = AllControlModels[control_model_name]["model"](control_model_params, env, device)

    logger = Logger(params.JobParams)

    logger.log_all_params(all_params=params, mode=params.JobParams.run_mode)

    if params.PolicyParams.ControlModuleParams.pretrained_model_path is not None:
        agent.load_model(params.PolicyParams.ControlModuleParams.pretrained_model_path)

    if params.JobParams.run_mode == "train":
        train_agent(params, agent, env, logger)

    elif params.JobParams.run_mode == 'test':
        evaluate_agent(params, agent, env, logger)


def train_agent(params, agent, env, logger):
    agent_params = params.PolicyParams.ControlModuleParams.control_model_params
    ep = 0
    moving_acc_reward = 0
    best_reward_score = None

    while logger.global_steps <= agent_params.total_timesteps:

        obs, _ = env.reset()
        rgb, _, _, _ = env.get_vision_observation()
        rgb_observation = rgb[:, :, :3]
        logger.reset_log_info()
        terminations = False

        for step in range(agent_params.max_episodic_timesteps):

            logger.global_steps += 1

            if logger.global_steps < agent_params.learning_starts:
                actions = np.array(env.action_space.sample())  # here try to use environment vector
            else:
                actions = agent.get_actions(obs, mode='train')

            next_obs, rewards, terminations, _, infos = env.step(actions)

            rgb_next, _, _, _ = env.get_vision_observation()
            rgb_observation_next = rgb_next[:, :, :3]

            real_next_obs = next_obs.copy()
            real_rgb_obs_next = rgb_observation_next.copy()

            agent.replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

            obs = next_obs
            rgb_observation = rgb_observation_next

            logger.log_step_performance_info(rewards, env.unwrapped.get_states_robot())
            if logger.global_steps > agent_params.learning_starts:
                loss_dict = agent.train_step(logger.global_steps)
                logger.log_step_loss_info(loss_dict)

            if terminations:
                break

        logger.log_run_results(ep, terminations, 'train')

        if ep % params.JobParams.evaluation_period == 0:
            acc_reward = evaluate_agent(params, agent, env, logger, ep)
            if best_reward_score is None:
                best_reward_score = acc_reward
            moving_acc_reward = 0.95 * moving_acc_reward + 0.05 * acc_reward

            if moving_acc_reward > best_reward_score:
                agent.save_model(logger.model_dir + 'best_model')
                best_reward_score = moving_acc_reward

        ep += 1

        agent.save_model(logger.model_dir + 'latest_model')

    agent.save_model(logger.model_dir + 'final_model')


def evaluate_agent(params, agent, env, logger, ep=0):
    agent_params = params.PolicyParams.ControlModuleParams.control_model_params
    obs, _ = env.reset()
    logger.reset_log_info()
    terminations = False

    for step in range(agent_params.max_episodic_timesteps):

        actions = agent.get_actions(obs, mode='eval')
        next_obs, rewards, terminations, _, infos = env.step(actions)
        obs = next_obs
        logger.log_step_performance_info(rewards, env.unwrapped.get_states_robot())
        if terminations:
            break

    logger.log_run_results(ep, terminations, mode='eval')
    acc_reward = logger.episode_accumulated_reward
    return acc_reward


if __name__ == '__main__':
    run()
