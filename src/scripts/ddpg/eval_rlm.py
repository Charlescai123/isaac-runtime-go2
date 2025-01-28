"""Evaluate a trained policy."""
import warnings

"""
python -m src.scripts.ddpg.eval_rlm --logdir=logs/train/ddpg_trot/demo --num_envs=1 --use_gpu=False --show_gui=True --use_real_robot=False --save_traj=True
"""

from absl import app
from absl import flags
from isaacgym import gymapi, gymutil
from datetime import datetime
import os
import pickle
import time

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import torch
import yaml

from isaacgym.terrain_utils import *
from src.envs import env_wrappers

torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_string("traj_dir", "logs/eval/", "traj_dir.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
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
    # print(f"flag: {FLAGS.save_traj}")
    # time.sleep(123)

    yaml_file_path = "src/configs/ddpg.yaml"

    with open(yaml_file_path, 'r') as file:
        cfg = yaml.safe_load(file)
    # from types import SimpleNamespace
    # cfg = SimpleNamespace(**cfg_dict)
    print(cfg)
    # time.sleep(123)

    from src.scripts.ppo.ddpg_agent import DDPGAgent
    ddpg_agent = DDPGAgent(cfg['agents'])

    device = "cuda" if FLAGS.use_gpu else "cpu"

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

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    env = config.env_class(num_envs=FLAGS.num_envs,
                           device=device,
                           config=config.environment,
                           show_gui=FLAGS.show_gui,
                           use_real_robot=FLAGS.use_real_robot)
    # add_uneven_terrains(gym=gym, sim=sim)
    env = env_wrappers.RangeNormalize(env)
    if FLAGS.use_real_robot:
        env.robot.state_estimator.use_external_contact_estimator = (not FLAGS.use_contact_sensor)

    # Retrieve policy
    # runner = OnPolicyRunner(env, config.training, policy_path, device=device)
    # runner.load(policy_path)
    # policy = runner.get_inference_policy()
    # runner.alg.actor_critic.train()

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

    env._torque_optimizer._base_position_kp *= 1
    env._torque_optimizer._base_position_kd *= 1
    env._torque_optimizer._base_orientation_kd *= 1
    env._torque_optimizer._base_orientation_kp *= 1
    # env._swing_leg_controller._foot_landing_clearance = 0.1    # current 0
    env._swing_leg_controller._foot_height = 0.12  # current 0.1

    print(f"swing: {env._swing_leg_controller._foot_landing_clearance}")
    print(f"swing: {env._swing_leg_controller._desired_base_height}")
    print(f"swing: {env._swing_leg_controller._foot_height}")
    # time.sleep(123)
    print(f"robot: {env._torque_optimizer._base_position_kp}")
    print(f"robot: {env._torque_optimizer._base_position_kd}")
    print(f"robot: {env._torque_optimizer._base_orientation_kp}")
    print(f"robot: {env._torque_optimizer._base_orientation_kd}")

    # time.sleep(1)
    start_time = time.time()
    logs = []
    with torch.inference_mode():
        while True:
            s = time.time()
            steps_count += 1
            # time.sleep(0.02)
            print(f"state is: {state}")

            # action = policy(state)

            def add_beta_noise(action):
                np.random.seed(1)
                action = action.cpu().numpy()
                beta_distribution_noise = np.random.beta(a=1.5, b=0.8, size=6) * 20
                action += beta_distribution_noise
                # action = np.clip(action, -1.0, 1.0)
                return to_torch(action, device=device)

            # time.sleep(123)
            # print(f"state is: {action2}")
            # print(f"state is: {type(action2)}")

            # Original A1 Policy
            action = to_torch(ddpg_agent.actor(state.cpu().numpy()).numpy(), device=device)

            # Add beta noise
            print(f"pre action is: {action}")
            action = add_beta_noise(action=action)
            print(f"action is: {action}")

            print(f"action is: {action}")
            # print(f"action is: {type(action)}")
            # print(f"action is: {to_torch(action.numpy())}")
            # print(f"action is: {type(to_torch(action))}")
            # action = torch.zeros(6).unsqueeze(dim=0)
            state, _, reward, done, info = env.step(action)
            print(f"Time: {env.robot.time_since_reset}, Reward: {reward}")

            total_reward += reward
            logs.extend(info["logs"])
            # if done.any():
            #     print(info["episode"])
            #     break
            if steps_count == 280:
                break
            print(f"steps_count: {steps_count}")
            e = time.time()
            print(
                f"***********************************************************************************duration: {e - s}")

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
