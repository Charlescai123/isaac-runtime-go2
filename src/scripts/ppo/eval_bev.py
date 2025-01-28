"""Evaluate a trained policy."""

"""
python -m src.agents.ppo.eval --logdir=logs/train/pronk_cajun --num_envs=1 --use_gpu=False --show_gui=True --use_real_robot=False --save_traj=True
"""

from absl import app
from absl import flags
# from absl import logging
from datetime import datetime
import os
import pickle
import time

from isaacgym.terrain_utils import *
from src.envs import env_wrappers
from src.envs.robots.modules.sensor.bev_utils import *
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

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
    env = env_wrappers.RangeNormalize(env)
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
    env._swing_leg_controller._foot_height = 0.15  # current 0.1

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
            # pcd_points = np.load("new_pcld.npy")
            # pcd_points = pcd_points[:, [2, 0, 1]]
            # pcd_points[:, 0] += 10
            # pcd_points[:, 1] -= 30
            # pcd_points[:, 2] += 2
            # bev_img = birds_eye_point_cloud(pcd_points)
            # plt.imshow(bev_img)
            # plt.show()
            # print("loaded!!!!")
            # time.sleep(123)
            # Add BEV
            # add_bev_map(env=env, idx=steps_count)
            print(f"steps_count: {steps_count}")
            print(f"Time: {env.robot.time_since_reset}, Reward: {reward}")

            total_reward += reward
            logs.extend(info["logs"])
            # if done.any():
            #     print(info["episode"])
            #     break
            if steps_count == 1000:
                break
            print(f"steps_count: {steps_count}")
            e = time.time()
            print(
                f"***********************************************************************************duration: {e - s}")
    if env.robot.record_video:
        record_depth_video(env._robot._frames)

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


def record_rgb_video(frames):
    import cv2
    frame_width = 1920
    frame_height = 1080
    fps = 30
    output_filename = 'camera_rgb.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    for frame in frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def record_depth_video(frames):
    import cv2
    frame_width = 1920
    frame_height = 1080
    fps = 30
    output_filename = 'camera_depth.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    max_depth = 10.0
    for depth_map in frames:
        # 1. 将深度图归一化到 [0, 255] 并转换为 uint8
        depth_normalized = (depth_map / max_depth * 255).astype(np.uint8)

        # 2. 可选：使用伪彩色增强可视化
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 3. 写入视频
        video_writer.write(depth_colored)

    video_writer.release()
    cv2.destroyAllWindows()


def update_depth_buffer(self):
    from isaacgym import gymtorch
    if not self.cfg.depth.use_camera:
        return

    if self.global_counter % self.cfg.depth.update_interval != 0:
        return
    self.gym.step_graphics(self.sim)  # required to render in headless mode
    self.gym.render_all_camera_sensors(self.sim)
    self.gym.start_access_image_tensors(self.sim)

    for i in range(self.num_envs):
        depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
                                                            self.envs[i],
                                                            self.cam_handles[i],
                                                            gymapi.IMAGE_DEPTH)

        depth_image = gymtorch.wrap_tensor(depth_image_)
        depth_image = self.process_depth_image(depth_image, i)

        init_flag = self.episode_length_buf <= 1
        if init_flag[i]:
            self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
        else:
            self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
                                             dim=0)

    self.gym.end_access_image_tensors(self.sim)


if __name__ == "__main__":
    app.run(main)
