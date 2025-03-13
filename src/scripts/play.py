"""Evaluate a trained policy."""
import logging
import warnings

import cv2
from matplotlib import animation

from src.envs.robots.modules.planner.path_planner import PathPlanner
from src.utils.utils import energy_value_2d, ActionMode
from src.physical_design import MATRIX_P

"""
python -m src.scripts.ddpg.play --logdir=logs/train/ddpg_trot/demo --use_gpu=True --enable_ha_teacher=True
"""
import os
import pickle
import time
from absl import app
from absl import flags
# from absl import logging
from isaacgym import gymapi, gymutil
from src.configs.training import ddpg
from datetime import datetime
from src.configs.defaults import sim_config
from src.envs.robots.modules.controller import raibert_swing_leg_controller, qp_torque_optimizer
from src.envs.robots.modules.gait_generator import phase_gait_generator
from src.envs.robots.motors import MotorControlMode
from src.envs.robots import go2_robot, go2
from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from rsl_rl.runners import OffPolicyRunner
import numpy as np
import torch
import yaml
import ml_collections
from tqdm import tqdm
from isaacgym.terrain_utils import *
from src.envs.terrains.wild_terrain_env import WildTerrainEnv
from src.envs import env_wrappers

torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_string("traj_dir", "logs/play/", "traj_dir.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("enable_ha_teacher", False, "whether to enable the HA-Teacher.")
flags.DEFINE_bool("enable_pusher", False, "whether to enable the robot pusher.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1, "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", True, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")
flags.DEFINE_float("total_time_secs", 10., "total amount of time to run the controller.")
# config_flags.DEFINE_config_file("config", "src/configs/wild_env_config.py", "experiment configuration.")

FLAGS = flags.FLAGS


def create_sim(sim_conf):
    gym = gymapi.acquire_gym()
    _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
    if sim_conf.show_gui:
        graphics_device_id = sim_device_id
    else:
        graphics_device_id = -1

    sim = gym.create_sim(sim_device_id, graphics_device_id,
                         sim_conf.physics_engine, sim_conf.sim_params)

    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V,
                                            "toggle_viewer_sync")
    else:
        viewer = None

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.static_friction = 0.
    plane_params.dynamic_friction = 0.
    plane_params.restitution = 0.
    gym.add_ground(sim, plane_params)
    return sim, viewer


def get_init_positions(num_envs: int,
                       distance: float = 1.,
                       device: str = "cpu") -> torch.Tensor:
    num_cols = int(np.sqrt(num_envs))
    init_positions = np.zeros((num_envs, 3))
    for idx in range(num_envs):
        init_positions[idx, 0] = idx // num_cols * distance
        init_positions[idx, 1] = idx % num_cols * distance
        init_positions[idx, 2] = 0.3
    return to_torch(init_positions, device=device)


def get_gait_config():
    config = ml_collections.ConfigDict()
    config.stepping_frequency = 2  # 1
    config.initial_offset = np.array([0., 0.5, 0.5, 0.], dtype=np.float32) * (2 * np.pi)
    config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32) * 1.
    # config.initial_offset = np.array([0.15, 0.15, -0.35, -0.35]) * (2 * np.pi)
    # config.swing_ratio = np.array([0.6, 0.6, 0.6, 0.6])
    return config


def main(argv):
    del argv  # unused
    logging.disable(logging.CRITICAL)  # logging output
    device = 'cuda' if FLAGS.use_gpu else 'cpu'
    sim_conf = sim_config.get_config(use_gpu=FLAGS.use_gpu,
                                     show_gui=FLAGS.show_gui)
    sim, viewer = create_sim(sim_conf)

    if FLAGS.use_real_robot:
        robot_class = go2_robot.Go2Robot
    else:
        robot_class = go2.Go2

    ans = input("Any Key...")
    if ans in ['y', 'Y']:
        import pdb
        pdb.set_trace()

    robot = robot_class(num_envs=FLAGS.num_envs,
                        init_positions=get_init_positions(
                            FLAGS.num_envs, device=sim_conf.sim_device),
                        sim=sim,
                        world_env=WildTerrainEnv,
                        viewer=viewer,
                        sim_config=sim_conf,
                        motor_control_mode=MotorControlMode.HYBRID)

    mean_pos = torch.min(robot.base_position_world, dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
    # mean_pos = torch.min(self.base_position_world,
    #                      dim=0)[0].cpu().numpy() + np.array([0.5, -1., 0.])
    target_pos = torch.mean(robot.base_position_world, dim=0).cpu().numpy() + np.array([0., 2., -0.5])
    print(f"target_pos: {target_pos}")
    cam_pos = gymapi.Vec3(*mean_pos)
    cam_target = gymapi.Vec3(*target_pos)
    robot._gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    robot._gym.step_graphics(robot._sim)
    robot._gym.draw_viewer(viewer, robot._sim, True)

    logs = []
    gait_config = get_gait_config()
    gait_generator = phase_gait_generator.PhaseGaitGenerator(robot, gait_config)
    swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot, gait_generator, foot_landing_clearance=0.01, foot_height=0.12)
    torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
        robot,
        desired_body_height=0.3,
        weight_ddq=np.diag([1., 1., 1., 10., 10., 1.]),
        foot_friction_coef=0.7,
        use_full_qp=False,
        clip_grf=True
    )
    desired_vx = 0.4

    # Path Planner
    planner = PathPlanner(robot=robot,
                          sim=sim,
                          viewer=viewer,
                          num_envs=FLAGS.num_envs,
                          device=device)
    init_map_flag = True
    grace_period = 200
    grace_cnt = 0
    robot.reset()
    planner.reset()

    steps_count = 0
    start_time = time.time()
    pbar = tqdm(total=FLAGS.total_time_secs)
    with torch.inference_mode():
        while robot.time_since_reset[0] <= FLAGS.total_time_secs:
            s = time.time()
            if FLAGS.use_real_robot:
                robot.state_estimator.update_foot_contact(
                    gait_generator.desired_contact_state)  # pytype: disable=attribute-error

            gait_generator.update()
            swing_leg_controller.update()
            # robot.state_estimator.update_ground_normal_vec()
            # robot.state_estimator.update_foot_contact(gait_generator.desired_contact_state)

            # Let the planner makes planning
            if planner.planning_flag:

                # Do map initialize once
                if init_map_flag:
                    planner.init_map()
                    init_map_flag = False

                # Take the trajectory generated from the planner
                ut, sub_goal_reach_flag = planner.get_reference_trajectory()
                vel_x = ut[0] * 1
                ref_wz = ut[1] * 1

                # Set desired command for vx and wz
                # self.desired_vx = vel_x
                desired_wz = np.clip(ref_wz, a_min=-0.65, a_max=0.65)

            else:
                if grace_cnt < grace_period:
                    time.sleep(0.02)
                    desired_vx = 0.
                    desired_wz = 0.
                    grace_cnt += 1
                else:
                    break
            # Update trajectory command
            torque_optimizer.set_controller_reference(
                desired_height=0.3,
                desired_lin_vel=[desired_vx, 0, 0],
                desired_rpy=[0, 0, 0],
                desired_ang_vel=[0, 0, desired_wz]
            )

            desired_acc, solved_acc, qp_cost, num_clips = torque_optimizer.compute_model_acc(
                foot_contact_state=gait_generator.desired_contact_state,
                desired_foot_position=swing_leg_controller.desired_foot_positions
            )

            motor_action, desired_acc, solved_acc, _, _ = torque_optimizer.get_action(
                foot_contact_state=gait_generator.desired_contact_state,
                swing_foot_position=swing_leg_controller.desired_foot_positions,
                generated_acc=desired_acc,
                gravity_frame=True
            )
            e = time.time()
            print(f"duration is: {e - s}")

            energy_2d = energy_value_2d(state=torque_optimizer.tracking_error[:, 2:],
                                        p_mat=to_torch(MATRIX_P, device=device))

            logs.append(
                dict(timestamp=robot.time_since_reset,
                     base_position=torch.clone(robot.base_position),
                     base_orientation_rpy=torch.clone(robot.base_orientation_rpy),
                     base_velocity=torch.clone(robot.base_velocity_body_frame),
                     base_angular_velocity=torch.clone(robot.base_angular_velocity_body_frame),
                     motor_positions=torch.clone(robot.motor_positions),
                     motor_velocities=torch.clone(robot.motor_velocities),
                     motor_action=motor_action,
                     motor_torques=robot.motor_torques,
                     motor_power=robot.motor_power,
                     num_clips=num_clips,
                     foot_contact_state=gait_generator.desired_contact_state,
                     foot_contact_force=robot.foot_contact_forces,
                     desired_swing_foot_position=swing_leg_controller.desired_foot_positions,
                     desired_acc_body_frame=desired_acc,
                     desired_vx=torque_optimizer.desired_linear_velocity[:, 0],
                     desired_wz=desired_wz,
                     desired_com_height=torque_optimizer.desired_base_position[:, 2].cpu(),
                     ha_action=desired_acc,
                     hp_action=desired_acc,
                     action_mode=torch.full((FLAGS.num_envs,), ActionMode.STUDENT.value, dtype=torch.int64,
                                            device=device),
                     acc_min=to_torch([-10, -10, -10, -20, -20, -20], device=sim_conf.sim_device),
                     acc_max=to_torch([10, 10, 10, 20, 20, 20], device=sim_conf.sim_device),
                     lyapunov_energy=to_torch(energy_2d, device=sim_conf.sim_device),
                     solved_acc_body_frame=solved_acc,
                     foot_positions_in_base_frame=robot.foot_positions_in_base_frame,
                     # env_action=drl_action,
                     # env_obs=torch.clone(self._obs_buf)
                     ))

            robot.step(motor_action)

            steps_count += 1
            # time.sleep(0.2)
            pbar.update(0.002)
            robot.render()

        print("Wallclock time: {}".format(time.time() - start_time))

        if FLAGS.save_traj:
            mode = "real" if FLAGS.use_real_robot else "sim"
            output_dir = f"eval_{mode}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl"
            # output_path = os.path.join(root_path, output_dir)
            output_path = os.path.join(os.path.dirname(FLAGS.traj_dir), output_dir)
            os.makedirs(FLAGS.traj_dir, exist_ok=True)  # Make sure the path exists

            with open(output_path, "wb") as fh:
                pickle.dump(logs, fh)
            print(f"Data logged to: {output_path}")


        import pdb
        pdb.set_trace()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("output.avi", fourcc, 30, (480, 640))
        out2 = cv2.VideoWriter("output2.avi", fourcc, 30, (480, 640))
        for i in range(len(robot._rgb_frames)):
            out.write(robot._rgb_frames[i])
            # out2.write(robot._dep_frames[i])
        out.release()
        out2.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(main)
