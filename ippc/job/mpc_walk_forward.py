import os
import sys
import click
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad_gym.env_builder import build_a1_ground_env
from quad_gym.gym_config import GymConfig
from utils.utils import load_config, save_params
from policy.control.mpc.MPC_Control import MPC, MPCConfig


@click.command()
@click.argument('config')
@click.option('--gpu', is_flag=True)
@click.option('--generate', is_flag=True, help="Generate a config file instead.")
@click.option('-p', '--params', nargs=2, multiple=True)
def run(**kwargs):
    if kwargs['generate']:
        save_params(GymConfig(), kwargs['config'])
        print("Config file generated.")
        return

    if not kwargs['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params: GymConfig = load_config(kwargs['config'], overrides=kwargs['params'])
    env = build_a1_ground_env(params)
    mpc = MPC(MPCConfig(), env.robot)

    for i in range(10):
        print("reset")
        current_time = env.robot.GetTimeSinceReset()
        env.reset()
        mpc.reset(env.robot)
        for j in range(10000):
            lin_speed, ang_speed = mpc.generate_example_linear_angular_speed(current_time)
            lin_speed = (0.5, 0., 0.)
            mpc.update_controller_params(lin_speed, ang_speed)
            mpc.controller.update()
            action, info = mpc.controller.get_action()
            if params.RobotParams.motor_control_mode == 'torque':
                action = mpc.convert2torque(action, env.robot)
            _, _, ter, trunc, _ = env.step(action)
            current_time = env.robot.GetTimeSinceReset()


if __name__ == '__main__':
    run()
