"""Evaluate a trained policy."""
import logging
import warnings

"""
python -m src.scripts.ddpg.eval --logdir=logs/train/ddpg_trot/demo --num_envs=1 --use_gpu=True --enable_ha_teacher=True
"""

from absl import app
from absl import flags
# from absl import logging
from isaacgym import gymapi, gymutil
from src.configs.training import ddpg
from datetime import datetime
import os
import pickle
import time

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from rsl_rl.runners import OffPolicyRunner
import numpy as np
import torch
import yaml

from isaacgym.terrain_utils import *
from src.envs import env_wrappers

torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_string("traj_dir", "logs/eval/", "traj_dir.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("enable_ha_teacher", False, "whether to enable the HA-Teacher.")
flags.DEFINE_bool("enable_pusher", False, "whether to enable the robot pusher.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1,
                     "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", True, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
    files = [
        entry for entry in os.listdir(logdir)
        if os.path.isfile(os.path.join(logdir, entry))
    ]
    files.sort(key=lambda entry: os.path.getmtime(os.path.join(logdir, entry)))
    files = files[::-1]

    for entry in files:
        if entry.startswith("model"):
            return os.path.join(logdir, entry)
    raise ValueError("No Valid Policy Found.")


def main(argv):
    del argv  # unused
    logging.disable(logging.CRITICAL)  # logging output

    # Load config and policy
    if FLAGS.logdir.endswith("pt"):
        config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
        policy_path = FLAGS.logdir
        root_path = os.path.dirname(FLAGS.logdir)
    else:
        # Find the latest policy ckpt
        config_path = os.path.join(FLAGS.logdir, "config.yaml")
        policy_path = get_latest_policy_path(FLAGS.logdir)
        root_path = FLAGS.logdir

    device = "cuda" if FLAGS.use_gpu else "cpu"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Use DDPG
    config.training = ddpg.get_training_config()

    # Reconfigure for plotting
    with config.unlocked():
        config.environment.terminate_on_destination_reach = True        # For video recording

        # config.environment.gamma = 0.0001
        config.environment.desired_vx = 0.4
        # config.environment.swing_foot_height = 0.11
        # config.environment.clip_wz = [-0.7, 0.7]
        # to_torch([-30, -30, -10, -20, -20, -20], device=self._device),
        # to_torch([30, 30, 30, 20, 20, 20], device=self._device))
        # config.environment.action_lb = np.array([-5., -10., -5., -5., -5., -4.])
        # config.environment.action_ub = np.array([10., 10., 5., 5., 5., 4.])
        # config.environment.action_lb = np.array([-10., -10., -10., -10., -10., -10.])
        # config.environment.action_ub = np.array([10., 10., 10., 10., 10., 10.])

        # config.environment.base_position_kp = np.array([0., 0., 50.])
        # config.environment.base_position_kd = np.array([10., 10., 10.])
        # config.environment.base_orientation_kp = np.array([50., 50., 0.])
        # config.environment.base_orientation_kd = np.array([10., 10., 10.])

        config.environment.terminate_on_dense_body_contact = False
        config.environment.terminate_on_height = 0.15

    # HA-Teacher module
    if FLAGS.enable_ha_teacher:
        with config.unlocked():
            # config.environment.safety_subset = [0.1, 0.28, 0.28, np.inf, 0.33, np.inf, np.inf, np.inf, np.inf, 1.]
            config.environment.ha_teacher.enable = True
            config.environment.ha_teacher.chi = 0.2
            config.environment.ha_teacher.tau = 20

    env = config.env_class(num_envs=FLAGS.num_envs,
                           device=device,
                           config=config.environment,
                           show_gui=FLAGS.show_gui,
                           use_real_robot=FLAGS.use_real_robot)

    # Robot pusher
    if FLAGS.enable_pusher:
        env._pusher.push_enable = True

    env = env_wrappers.RangeNormalize(env)
    if FLAGS.use_real_robot:
        env.robot.state_estimator.use_external_contact_estimator = (not FLAGS.use_contact_sensor)

    # Retrieve policy
    runner = OffPolicyRunner(env, config.training, policy_path, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy()

    # Reset environment
    state, _ = env.reset()
    total_reward = torch.zeros(FLAGS.num_envs, device=device)
    steps_count = 0

    mean_pos = torch.min(env.robot.base_position_world, dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
    # mean_pos = torch.min(self.base_position_world,
    #                      dim=0)[0].cpu().numpy() + np.array([0.5, -1., 0.])
    target_pos = torch.mean(env.robot.base_position_world, dim=0).cpu().numpy() + np.array([0., 2., -0.5])
    cam_pos = gymapi.Vec3(*mean_pos)
    cam_target = gymapi.Vec3(*target_pos)
    env.robot._gym.viewer_camera_look_at(env.robot._viewer, None, cam_pos, cam_target)
    env.robot._gym.step_graphics(env.robot._sim)
    env.robot._gym.draw_viewer(env.robot._viewer, env.robot._sim, True)

    # env._torque_optimizer._base_position_kp *= 0.5
    # env._torque_optimizer._base_position_kd *= 0.5
    # env._torque_optimizer._base_orientation_kp *= 0.5
    # env._torque_optimizer._base_orientation_kd *= 0.5

    print(f"env._torque_optimizer._base_position_kp: {env._torque_optimizer._base_position_kp}")
    print(f"env._torque_optimizer._base_position_kd: {env._torque_optimizer._base_position_kd}")
    print(f"env._torque_optimizer._base_orientation_kp: {env._torque_optimizer._base_orientation_kp}")
    print(f"env._torque_optimizer._base_orientation_kd: {env._torque_optimizer._base_orientation_kd}")
    # time.sleep(123)
    start_time = time.time()
    logs = []
    wait_step = 0
    wait_buffer = 400
    done_flag = False
    with torch.inference_mode():
        while True:
            s = time.time()
            steps_count += 1
            # time.sleep(0.02)
            print(f"state is: {state}")

            action = policy(state)

            # action = torch.zeros([1, 6], device=device)

            def add_beta_noise(action):
                np.random.seed(1)
                action = action.cpu().numpy()
                beta_distribution_noise = np.random.beta(a=1.5, b=0.8, size=6) * 20
                action += beta_distribution_noise
                # action = np.clip(action, -1.0, 1.0)
                return to_torch(action, device=device)

            # Add beta noise
            print(f"pre action is: {action}")
            # action = add_beta_noise(action=action)
            print(f"action after adding noise is: {action}")

            # print(f"action is: {type(action)}")
            # print(f"action is: {to_torch(action.numpy())}")
            # print(f"action is: {type(to_torch(action))}")
            # action = torch.zeros(6).unsqueeze(dim=0)
            state, _, nominal_action, reward, done, info = env.step(action)
            print(f"Time: {env.robot.time_since_reset}, Reward: {reward}")

            total_reward += reward
            logs.extend(info["logs"])

            # import pdb
            # pdb.set_trace()

            if info["fails"] >= 1:
                # import pdb
                # pdb.set_trace()
                done_flag = True

            if steps_count == 4500 or done.any():
            # if steps_count == 1000 or done.any():
                print(info["episode"])
                done_flag = True
                break

            if done_flag:
                if wait_step < wait_buffer:
                    wait_step += 1
                else:
                    break

            if env.planner.planning_flag is False:
                if wait_step < wait_buffer:
                    wait_step += 1
                else:
                    break
            print(f"steps_count: {steps_count}")
            e = time.time()
            print(f"step duration: {e - s}")
            print(f"wait_step: {wait_step}")

    print(f"Total reward: {total_reward}")
    print(f"Time elapsed: {time.time() - start_time}")
    if FLAGS.use_real_robot or FLAGS.save_traj:
        mode = "real" if FLAGS.use_real_robot else "sim"
        output_dir = (
            f"eval_{mode}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl")
        # output_path = os.path.join(root_path, output_dir)
        output_path = os.path.join(os.path.dirname(FLAGS.traj_dir), output_dir)

        with open(output_path, "wb") as fh:
            pickle.dump(logs, fh)
        print(f"Data logged to: {output_path}")


if __name__ == "__main__":
    app.run(main)
