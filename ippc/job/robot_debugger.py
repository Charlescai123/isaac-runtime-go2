"""Simple script for executing random actions on A1 robot."""

import numpy as np
from tqdm import tqdm
import pybullet as p  # pytype: disable=import-error


import dataclasses
import os
import sys
import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad_gym.env_builder import build_a1_ground_env
from quad_gym.gym_config import GymConfig
from utils.utils import load_config
from policy.policy_config import PolicyConfig
from job.job_config import JobConfig
from dataclasses_json import dataclass_json


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
    params: AllConfig = load_config(kwargs['config'], overrides=kwargs['params'])
    params.GymParams.SimParams.enable_rendering = True
    env = build_a1_ground_env(params.GymParams)

    action_low, action_high = env.action_space.low, env.action_space.high
    action_median = (action_low + action_high) / 2.
    dim_action = action_low.shape[0]
    action_selector_ids = []
    for dim in range(dim_action):
        action_selector_id = p.addUserDebugParameter(paramName='dim{}'.format(dim),
                                                     rangeMin=action_low[dim],
                                                     rangeMax=action_high[dim],
                                                     startValue=action_median[dim])
        action_selector_ids.append(action_selector_id)

    for _ in tqdm(range(800)):
        action = np.zeros(dim_action)
        for dim in range(dim_action):
            action[dim] = env.pybullet_client.readUserDebugParameter(
                action_selector_ids[dim])
        env.step(action)


if __name__ == "__main__":
    run()
