import os
import sys
import click

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quad_gym.env_builder import build_a1_ground_env
from quad_gym.gym_config import GymConfig
from policy.control.mpc.MPC_Control import MPC, MPCConfig
from utils.utils import load_config, save_params
from PIL import Image
import numpy as np


@click.command()
@click.argument('config')
@click.option('--gpu', is_flag=True)
@click.option('--generate', is_flag=True, help="Generate a config file instead.")
@click.option('-p', '--params', nargs=2, multiple=True)
def gen_data(**kwargs):
    print(kwargs)
    if kwargs['generate']:
        save_params(GymConfig(), kwargs['config'])
        print("Config file generated.")
        return

    if not kwargs['gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    params: GymConfig = load_config(kwargs['config'], overrides=kwargs['params'])
    rgb_save_folder = './data/perception/bev/rgb'
    depth_save_folder = './data/perception/bev/depth'
    seg_save_folder = './data/perception/bev/seg'
    label_save_folder = './data/perception/bev/label'

    os.makedirs(rgb_save_folder, exist_ok=True)
    os.makedirs(depth_save_folder, exist_ok=True)
    os.makedirs(seg_save_folder, exist_ok=True)
    os.makedirs(label_save_folder, exist_ok=True)

    env = build_a1_ground_env(params.GymParams)
    mpc = MPC(MPCConfig(), env.robot)
    num_episodes = 100
    save_data = True
    img_id = 0

    for i in range(num_episodes):
        print("reset")
        object_label = env.object_dict_list
        np.save(f"./data/perception/bev/scene_object_label.npy", object_label, allow_pickle=True)
        env.reset()
        mpc.reset(env.robot)
        current_time = env.robot.GetTimeSinceReset()
        for j in range(1000):
            lin_speed, ang_speed = mpc.generate_example_linear_angular_speed(current_time)
            lin_speed = (0.5, 0., 0.)
            mpc.update_controller_params(lin_speed, ang_speed)
            mpc.controller.update()
            action, info = mpc.controller.get_action()
            if params.GymParams.RobotParams.motor_control_mode == 'torque':
                action = mpc.convert2torque(action, env.robot)
            _, _, ter, trunc, _ = env.step(action)
            current_time = env.robot.GetTimeSinceReset()
            rgb, depth, seg, info = env.get_vision_observation(return_label=True)

            if save_data:
                rgb_array = np.array(rgb)
                rgb_array = rgb_array[:, :, :3]
                im = Image.fromarray(rgb_array)
                depth = np.array(depth)
                rgb_save_path = os.path.join(rgb_save_folder, f"rgb_{img_id}.png")
                depth_save_path = os.path.join(depth_save_folder, f"depth_{img_id}.npy")
                seg_save_path = os.path.join(seg_save_folder, f"seg_{img_id}.npy")
                label_save_path = os.path.join(label_save_folder, f"label_{img_id}.npy")

                im.save(rgb_save_path)
                np.save(seg_save_path, seg, allow_pickle=True)
                np.save(depth_save_path, depth, allow_pickle=True)
                np.save(label_save_path, info, allow_pickle=True)
                img_id += 1

            if ter:
                break

        print("finish eposide")


if __name__ == '__main__':
    gen_data()
