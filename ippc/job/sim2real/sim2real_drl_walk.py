import dataclasses
import os
import sys
import click
import random

import numpy as np
import torch
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


from quad_A1.control_loop_execution.rl_policy_state_wrapper import PolicyWrapper
from quad_A1.control_loop_execution.main_executor_state import Executor
from quad_A1.a1_utilities.robot_controller import RobotController
from quad_A1.a1_utilities.realsense import A1RealSense
from quad_A1.a1_utilities.a1_sensor_process import *


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
    robot = env.robot

    control_model_name = params.PolicyParams.ControlModuleParams.control_model
    control_model_params = params.PolicyParams.ControlModuleParams.control_model_params

    agent = AllControlModels[control_model_name]["model"](control_model_params, env, device)

    logger = Logger(params.JobParams)

    logger.log_all_params(all_params=params, mode=params.JobParams.run_mode)

    if params.PolicyParams.ControlModuleParams.pretrained_model_path is not None:
        agent.load_model(params.PolicyParams.ControlModuleParams.pretrained_model_path)

    save_dir_name = logger.log_dir
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)

    robot_controller = RobotController(state_save_path=save_dir_name)
    realsense = A1RealSense(save_dir_name=save_dir_name)

    # NORM_PATH = ''

    # with open(NORM_PATH, 'rb') as f:
    #     obs_normalizer = pickle.load(f)

    # obs_normalizer_mean = obs_normalizer._mean
    # obs_normalizer_var = obs_normalizer._var

    obs_normalizer_mean = np.zeros(env.observation_space.shape[0])
    obs_normalizer_var = np.ones(env.observation_space.shape[0])

    num_action_repeat = params.GymParams.SimParams.num_action_repeat
    control_frequency = 33

    print("Warm Up")
    for i in range(100):
        f_input = np.random.rand(env.observation_space.shape[0])
        _ = agent.get_actions(f_input, mode='eval')
    print("Warm Up done")

    policyComputer = PolicyWrapper(
        robot,
        params.GymParams,
        agent,
        obs_normalizer_mean,
        obs_normalizer_var,
        save_dir_name=save_dir_name,
        no_tensor=True,
        action_range=[0.05, 0.5, 0.5]
    )

    executor = Executor(
        realsense,
        robot_controller,
        policyComputer,
        control_freq=control_frequency,
        Kp=40, Kd=0.4
    )

    EXECUTION_TIME = 8
    executor.execute(EXECUTION_TIME)


if __name__ == '__main__':
    run()
