"""Evaluate a trained policy."""

"""
python -m src.scripts.ppo.eval --logdir=logs/train/pronk_cajun/2024_10_31_00_04_18 --num_envs=1 --use_gpu=False --show_gui=True --use_real_robot=False --save_traj=True
"""

from absl import app
from absl import flags
# from absl import logging
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
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
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

    with config.unlocked():
        config.environment.jumping_distance_schedule = [1., 0.3]
        # config.environment.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 6
        config.environment.max_jumps = 6

    env = config.env_class(num_envs=FLAGS.num_envs,
                           device=device,
                           config=config.environment,
                           show_gui=FLAGS.show_gui,
                           use_real_robot=FLAGS.use_real_robot)
    # add_uneven_terrains(gym=env.robot._gym, sim=env.robot._sim)
    env = env_wrappers.RangeNormalize(env)   # Normalize the environment
    if FLAGS.use_real_robot:
        env.robot.state_estimator.use_external_contact_estimator = (not FLAGS.use_contact_sensor)

    # Retrieve policy
    runner = OnPolicyRunner(env, config.training, policy_path, device=device)
    runner.load(policy_path)
    policy = runner.get_inference_policy()
    runner.alg.actor_critic.train()

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
    env._swing_leg_controller._foot_height = 0.15        # current 0.1

    print(f"swing: {env._swing_leg_controller._foot_landing_clearance}")
    print(f"swing: {env._swing_leg_controller._desired_base_height}")
    print(f"swing: {env._swing_leg_controller._foot_height}")
    # time.sleep(123)
    print(f"robot: {env._torque_optimizer._base_position_kp}")
    print(f"robot: {env._torque_optimizer._base_position_kd}")
    print(f"robot: {env._torque_optimizer._base_orientation_kp}")
    print(f"robot: {env._torque_optimizer._base_orientation_kd}")

    # env.load_plane_asset()
    # env.add_snow_road()
    # Add uneven terrains to show the patch strength
    # add_uneven_terrains(gym=env.robot._gym, sim=env.robot._sim)
    # from src.utils.sim_utils import add_terrain
    # add_terrain(env._robot._gym, env._robot._sim)

    # time.sleep(3)
    start_time = time.time()
    logs = []
    with torch.inference_mode():
        while True:
            s = time.time()
            steps_count += 1
            # time.sleep(0.05)
            action = policy(state)
            print(f"action is: {action}")
            # action = torch.zeros(6).unsqueeze(dim=0)
            state, _, reward, done, info = env.step(action)
            print(f"Time: {env.robot.time_since_reset}, Reward: {reward}")

            total_reward += reward
            logs.extend(info["logs"])
            # if done.any():
            #     print(info["episode"])
            #     break
            if steps_count == 2000:
                break
            print(f"steps_count: {steps_count}")
            e = time.time()
            print(
                f"************************************duration: {e - s}")

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


def add_uneven_terrains(gym, sim):
    # terrains
    num_terrains = 4
    terrain_width = 12.
    terrain_length = 12.
    horizontal_scale = 0.1  # [m] resolution in x
    vertical_scale = 0.01  # [m] resolution in z
    num_rows = int(terrain_width / horizontal_scale)
    num_cols = int(terrain_length / horizontal_scale)
    heightfield = np.zeros((num_terrains * num_rows, num_cols), dtype=np.int16)

    def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale,
                                             horizontal_scale=horizontal_scale)

    # np.random.seed(42)  # works for vel 0.3 m/s
    np.random.seed(3)  # works for all vel
    heightfield[0:1 * num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.01, max_height=0.01,
                                                            step=0.05, downsampled_scale=0.1).height_field_raw
    heightfield[1 * num_rows:2 * num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
    heightfield[2 * num_rows:3 * num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75,
                                                               step_height=-0.35).height_field_raw
    heightfield[2 * num_rows:3 * num_rows, :] = heightfield[2 * num_rows:3 * num_rows, :][::-1]
    heightfield[3 * num_rows:4 * num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75,
                                                                       step_height=-0.5).height_field_raw

    # add the terrain as a triangle mesh
    vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale,
                                                         vertical_scale=vertical_scale, slope_threshold=1.5)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p.x = -12.
    tm_params.transform.p.y = -terrain_width / 2 - 1.
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)


if __name__ == "__main__":
    app.run(main)
