import dataclasses
import os
import sys
import click
import random
import torch
import numpy as np
import copy
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad_gym.env_builder import build_a1_ground_env
from quad_gym.gym_config import GymConfig
from utils.utils import load_config, save_params
from policy.policy_config import PolicyConfig
from job.job_config import JobConfig
from job.job_monitor.job_logger import Logger
from dataclasses_json import dataclass_json
from policy.policy_config import AllControlModels
from tqdm import tqdm

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

    interact_loop(params, agent, env, logger)


def interact_loop(params, agent, env, logger, ep=10):
    agent_params = params.PolicyParams.ControlModuleParams.control_model_params
    obs, _ = env.reset()
    logger.reset_log_info()
    terminations = False
    data = []

    for i in tqdm(range(ep)):
        obs_list = []
        actions_list = []
        rewards_list = []
        next_obs_list = []
        term_list = []

        for step in range(agent_params.num_steps*1):

            actions = agent.get_actions(obs, mode='eval')
            next_obs, rewards, terminations, _, infos = env.step(actions)

            term_list.append(copy.deepcopy(terminations))
            obs_list.append(copy.deepcopy(obs))
            actions_list.append(copy.deepcopy(actions))
            rewards_list.append(rewards)
            next_obs_list.append(copy.deepcopy(next_obs))

            obs = next_obs

            if terminations:
                break
        print("total_steps", len(rewards_list), "Mean_reward:", np.mean(rewards_list))
        data.append([np.array(obs_list), np.array(actions_list), np.array(rewards_list), np.array(next_obs_list)])

    with open("data/offline/simreal/generated_offline_dataset.dat", "wb") as f:
        pickle.dump(data, f)

    return


if __name__ == '__main__':
    run()
