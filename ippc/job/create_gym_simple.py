import os
import sys
import time
import click
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad_gym.env_builder import build_a1_ground_env
from quad_gym.gym_config import GymConfig
from utils.utils import load_config, save_params


@click.command()
@click.argument('config')
@click.option('--gpu', is_flag=True)
@click.option('--generate', is_flag=True, help="Generate a config file instead.")
@click.option('-p', '--params', nargs=2, multiple=True)

def test_env(**kwargs):
    print(kwargs)
    if kwargs['generate']:
        save_params(GymConfig(), kwargs['config'])
        print("Config file generated.")
        return

    if not kwargs['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params: GymConfig = load_config(kwargs['config'], overrides=kwargs['params'])

    env = build_a1_ground_env(params)
    print(env.action_space.low)
    print(env.action_space.high)

    c_t = time.time()
    for i in range(100000000):
        print("reset")
        env.reset()
        for j in range(1000):
            # print(env.action_space)
            _, r, ter, a, _ = env.step(env.action_space.sample())
            print("reward", r) # todo check reward function
            if ter:
                print("reset")
                env.reset()
    print(time.time() - c_t)
    print(env.count_t)
    print(10000 / (time.time() - c_t))


if __name__ == '__main__':
    test_env()
