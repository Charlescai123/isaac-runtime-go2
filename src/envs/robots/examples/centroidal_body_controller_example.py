"""Example of running the phase gait generator.
python -m src.envs.robots.examples.centroidal_body_controller_example  --num_envs=1 --use_gpu=False --show_gui=True --use_real_robot=False
"""
import os
import pickle
from datetime import datetime

from absl import app
from absl import flags

import time

from isaacgym.torch_utils import to_torch
import ml_collections
import scipy
from tqdm import tqdm
import torch

import matplotlib.pyplot as plt
from src.configs.defaults import sim_config
from src.envs.robots.modules.controller import raibert_swing_leg_controller, qp_torque_optimizer
from src.envs.robots.modules.gait_generator import phase_gait_generator
from src.envs.robots import go2_robot, go2
from src.envs.robots.motors import MotorControlMode
from isaacgym.terrain_utils import *
from src.envs.terrains.wild_env import WildTerrainEnv
from src.utils.utils import energy_value_2d, ActionMode
from src.physical_design import MATRIX_P

flags.DEFINE_integer("num_envs", 1, "number of environments to create.")
flags.DEFINE_float("total_time_secs", 100.,
                   "total amount of time to run the controller.")
flags.DEFINE_string("traj_dir", "logs/eval/", "traj_dir.")
flags.DEFINE_bool("use_gpu", True, "whether to show GUI.")
flags.DEFINE_bool("save_traj", True, "whether to save trajectory.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to run on real robot.")
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


def _generate_example_linear_angular_speed(t):
    """Creates an example speed profile based on time for demo purpose."""
    vx = 0.6
    vy = 0.2
    wz = 0.8

    # time_points = (0, 1, 9, 10, 15, 20, 25, 30)
    # speed_points = ((0, 0, 0, 0), (0, 0.6, 0, 0), (0, 0.6, 0, 0), (vx, 0, 0, 0),
    #                 (0, 0, 0, -wz), (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

    time_points = (0, 5, 10, 15, 20, 25, 30)
    speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
                    (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

    speed = scipy.interpolate.interp1d(time_points,
                                       speed_points,
                                       kind="nearest",
                                       fill_value="extrapolate",
                                       axis=0)(t)

    return [0.5, 0., 0.], 0.


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
    sim_conf = sim_config.get_config(use_gpu=FLAGS.use_gpu,
                                     show_gui=FLAGS.show_gui)
    sim, viewer = create_sim(sim_conf)

    if FLAGS.use_real_robot:
        robot_class = go2_robot.Go2Robot
    else:
        robot_class = go2.Go2

    robot = robot_class(num_envs=FLAGS.num_envs,
                        init_positions=get_init_positions(
                            FLAGS.num_envs, device=sim_conf.sim_device),
                        sim=sim,
                        world_env=WildTerrainEnv,
                        viewer=viewer,
                        sim_config=sim_conf,
                        motor_control_mode=MotorControlMode.HYBRID)

    from src.envs.terrains.sim_utils import add_terrain
    gym = gymapi.acquire_gym()

    # Slope and stairs
    offset_x = 49
    # add_terrain(gym, sim, "slope", offset_x)
    # add_terrain(gym, sim, "stair", offset_x + 3.95, True)

    mean_pos = torch.min(robot.base_position_world, dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
    # mean_pos = torch.min(self.base_position_world,
    #                      dim=0)[0].cpu().numpy() + np.array([0.5, -1., 0.])
    target_pos = torch.mean(robot.base_position_world, dim=0).cpu().numpy() + np.array([0., 2., -0.5])
    print(f"target_pos: {target_pos}")
    cam_pos = gymapi.Vec3(*mean_pos)
    cam_target = gymapi.Vec3(*target_pos)
    # robot._gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    # robot._gym.step_graphics(robot._sim)
    robot._gym.draw_viewer(viewer, robot._sim, True)

    # add_uneven_terrains(gym=robot._gym, sim=sim)
    # Add perception (BEV)
    camera_props = gymapi.CameraProperties()
    # camera_props.width = 512
    # camera_props.height = 512

    camera_props.enable_tensors = True

    print(f"robot._envs: {robot._envs}")

    gym = robot._gym

    # 将摄像机附加到环境中
    camera_properties = gymapi.CameraProperties()
    camera_handle = gym.create_camera_sensor(robot._envs[0], camera_properties)
    # gym.set_camera_location(camera_handle, robot._envs[0], camera_transform)

    # time.sleep(123)

    # camera_handle = robot._gym.create_camera_sensor(robot._envs[0], camera_props)
    mean_pos = torch.min(robot.base_position_world, dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
    target_pos = torch.mean(robot.base_position_world,
                            dim=0).cpu().numpy() + np.array([0., 2., -0.5])
    robot._gym.set_camera_location(camera_handle, robot._envs[0], gymapi.Vec3(*mean_pos), gymapi.Vec3(*target_pos))
    # robot._gym.render_all_camera_sensors(sim)
    f, axarr = plt.subplots(2, 2, figsize=(16, 16))
    plt.axis('off')
    plt.tight_layout(pad=0)

    logs = []
    # time.sleep(5)

    gait_config = get_gait_config()
    gait_generator = phase_gait_generator.PhaseGaitGenerator(robot, gait_config)
    swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot, gait_generator, foot_landing_clearance=0.01, foot_height=0.12)
    torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
        robot,
        desired_body_height=0.3,
        weight_ddq=np.diag([1., 1., 1., 10., 10., 1.]),
        foot_friction_coef=0.4,
        use_full_qp=False,
        clip_grf=True
    )

    robot.reset()
    num_envs, num_dof = robot.num_envs, robot.num_dof
    steps_count = 0
    torque_optimizer._base_position_kp *= 2
    torque_optimizer._base_position_kd *= 2
    torque_optimizer._base_orientation_kp *= 2
    torque_optimizer._base_orientation_kd *= 2
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

            # Update speed command
            lin_command, ang_command = _generate_example_linear_angular_speed(
                robot.time_since_reset[0].cpu())
            print(lin_command, ang_command)

            lin_vel = to_torch([0., 0, 0], device=sim_conf.sim_device)
            ang_vel = to_torch([0., 0, 0.5], device=sim_conf.sim_device)
            torque_optimizer.desired_linear_velocity = torch.stack([lin_vel] * robot.num_envs, dim=0)
            torque_optimizer.desired_angular_velocity = torch.stack([ang_vel] * robot.num_envs, dim=0)

            motor_action, desired_acc, solved_acc, _, _ = torque_optimizer.get_action(
                gait_generator.desired_contact_state,
                swing_foot_position=swing_leg_controller.desired_foot_positions)
            e = time.time()

            energy_2d = energy_value_2d(state=torque_optimizer.tracking_error[:, 2:], p_mat=MATRIX_P)

            print(f"torque_optimizer.tracking_error: {torque_optimizer.tracking_error}")
            print(f"robot: {torque_optimizer._base_position_kp}")
            print(f"robot: {torque_optimizer._base_position_kd}")
            print(f"robot: {torque_optimizer._base_orientation_kp}")
            print(f"robot: {torque_optimizer._base_orientation_kd}")
            print(f"duration is: {e - s}")

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
                     # num_clips=self._num_clips,
                     foot_contact_state=gait_generator.desired_contact_state,
                     foot_contact_force=robot.foot_contact_forces,
                     desired_swing_foot_position=swing_leg_controller.desired_foot_positions,
                     desired_acc_body_frame=desired_acc,
                     desired_vx=torque_optimizer.desired_linear_velocity[:, 0],
                     desired_com_height=torque_optimizer.desired_base_position[:, 2],
                     ha_action=desired_acc,
                     hp_action=desired_acc,
                     action_mode=ActionMode.TEACHER,
                     acc_min=to_torch([-10, -10, -10, -20, -20, -20], device=sim_conf.sim_device),
                     acc_max=to_torch([10, 10, 10, 20, 20, 20], device=sim_conf.sim_device),
                     energy=to_torch(energy_2d, device=sim_conf.sim_device),
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
    horizontal_scale = 0.05  # [m] resolution in x
    vertical_scale = 0.001  # [m] resolution in z
    num_rows = int(terrain_width / horizontal_scale)
    num_cols = int(terrain_length / horizontal_scale)
    heightfield = np.zeros((num_terrains * num_rows, num_cols), dtype=np.int16)

    def new_sub_terrain(): return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale,
                                             horizontal_scale=horizontal_scale)

    heightfield[0:1 * num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.01, max_height=0.01,
                                                            step=0.2, downsampled_scale=0.5).height_field_raw
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
    tm_params.transform.p.x = -1.
    tm_params.transform.p.y = -terrain_width / 2 - 1.
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)


if __name__ == "__main__":
    app.run(main)
