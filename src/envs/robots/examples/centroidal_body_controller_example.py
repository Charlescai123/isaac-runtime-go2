"""Example of running the phase gait generator.
python -m src.envs.robots.examples.centroidal_body_controller_example  --num_envs=2 --use_gpu=False --show_gui=True --use_real_robot=False
"""

from absl import app
from absl import flags

import time

from isaacgym.torch_utils import to_torch
import ml_collections
import scipy
from tqdm import tqdm
import torch

from src.configs.defaults import sim_config
from src.envs.robots.controller import raibert_swing_leg_controller, phase_gait_generator
from src.envs.robots.controller import qp_torque_optimizer
from src.envs.robots import go2_robot, go2
from src.envs.robots.motors import MotorControlMode
from isaacgym.terrain_utils import *
from src.envs.terrains.wild_env import WildTerrainEnv

flags.DEFINE_integer("num_envs", 1, "number of environments to create.")
flags.DEFINE_float("total_time_secs", 20.,
                   "total amount of time to run the controller.")
flags.DEFINE_bool("use_gpu", True, "whether to show GUI.")
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
    config.initial_offset = np.array([0., 0.5, 0.5, 0.],
                                     dtype=np.float32) * (2 * np.pi)
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

    mean_pos = torch.min(robot.base_position_world,
                         dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
    # mean_pos = torch.min(self.base_position_world,
    #                      dim=0)[0].cpu().numpy() + np.array([0.5, -1., 0.])
    target_pos = torch.mean(robot.base_position_world, dim=0).cpu().numpy() + np.array([0., 2., -0.5])
    print(f"target_pos: {target_pos}")
    cam_pos = gymapi.Vec3(*mean_pos)
    cam_target = gymapi.Vec3(*target_pos)
    # robot._gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    # robot._gym.step_graphics(robot._sim)
    robot._gym.draw_viewer(viewer, robot._sim, True)

    add_uneven_terrains(gym=robot._gym, sim=sim)
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
    import matplotlib.pyplot as plt
    # camera_handle = robot._gym.create_camera_sensor(robot._envs[0], camera_props)
    mean_pos = torch.min(robot.base_position_world,
                         dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
    target_pos = torch.mean(robot.base_position_world,
                            dim=0).cpu().numpy() + np.array([0., 2., -0.5])
    robot._gym.set_camera_location(camera_handle, robot._envs[0], gymapi.Vec3(*mean_pos), gymapi.Vec3(*target_pos))
    # robot._gym.render_all_camera_sensors(sim)
    f, axarr = plt.subplots(2, 2, figsize=(16, 16))
    plt.axis('off')
    plt.tight_layout(pad=0)
    stepX = 1
    stepY = 1
    near_val = 0.1
    far_val = 5

    # time.sleep(5)

    gait_config = get_gait_config()
    gait_generator = phase_gait_generator.PhaseGaitGenerator(robot, gait_config)
    swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot, gait_generator, foot_landing_clearance=0., foot_height=0.1)
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

            # Update speed command
            lin_command, ang_command = _generate_example_linear_angular_speed(
                robot.time_since_reset[0].cpu())
            # print(lin_command, ang_command)

            torque_optimizer.desired_linear_velocity = lin_command
            torque_optimizer.desired_angular_velocity = [0., 0., ang_command]

            motor_action, _, _, _, _ = torque_optimizer.get_action(
                gait_generator.desired_contact_state,
                swing_foot_position=swing_leg_controller.desired_foot_positions)
            e = time.time()


            print(f"torque_optimizer.tracking_error: {torque_optimizer.tracking_error}")
            print(f"robot: {torque_optimizer._base_position_kp}")
            print(f"robot: {torque_optimizer._base_position_kd}")
            print(f"robot: {torque_optimizer._base_orientation_kp}")
            print(f"robot: {torque_optimizer._base_orientation_kd}")

            print(f"duration is: {e - s}")
            robot.step(motor_action)

            steps_count += 1
            # time.sleep(0.2)
            pbar.update(0.002)
            robot.render()

            # Render BEV Perception
            def perception():
                rgb, dep, seg, info = get_vision_observation(gym=robot._gym, sim=sim, env=robot._envs[0],
                                                             camera_handle=camera_handle, i=steps_count,
                                                             return_label=True)
                mean_pos = torch.min(robot.base_position_world,
                                     dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
                target_pos = torch.mean(robot.base_position_world,
                                        dim=0).cpu().numpy() + np.array([0., 2., -0.5])
                robot._gym.set_camera_location(camera_handle, robot._envs[0], gymapi.Vec3(*mean_pos),
                                               gymapi.Vec3(*target_pos))
                # rgb, dep, seg, info = get_vision_observation_isaac(gym=robot._gym, sim=sim, env=None,
                #                                                    robot_handle=robot, return_label=True)

                projection_matrix = info["projection_matrix"]
                view_matrix = info["view_matrix"]
                imgW = info["width"]
                imgH = info["height"]
                # camPos = info["cam_pos"]

                # current_time = env.robot.GetTimeSinceReset()
                realDepthImg = dep.copy()

                for w in range(0, imgW, stepX):
                    for h in range(0, imgH, stepY):
                        def getDepth(z_n, zNear, zFar):
                            z_n = 2.0 * z_n - 1.0
                            z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear))
                            return z_e

                        realDepthImg[w][h] = getDepth(dep[w][h], near_val, far_val)

                # pointCloud = np.empty([np.int32(imgH / stepY), np.int32(imgW / stepX), 4])
                #
                # projectionMatrix = np.asarray(projection_matrix).reshape([4, 4], order='F')
                #
                # viewMatrix = np.asarray(view_matrix).reshape([4, 4], order='F')
                #
                # tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
                #
                # for h in range(0, imgH, stepY):
                #     for w in range(0, imgW, stepX):
                #         x = (2 * w - imgW) / imgW
                #         y = -(2 * h - imgH) / imgH  # be careful！ depth and its corresponding position
                #         z = 2 * dep[h, w] - 1
                #         # z = realDepthImg[h,w]
                #         pixPos = np.asarray([x, y, z, 1])
                #         # print(pixPos)
                #         position = np.matmul(tran_pix_world, pixPos)
                #         pointCloud[np.int32(h / stepY), np.int32(w / stepX), :] = position / position[3]

                # Here the point cloud is in the world frame
                # pointCloud = np.reshape(pointCloud[:, :, :3], newshape=(-1, 3))
                # # Transform the point cloud to the robot frame
                # # (we assume that the camera lies on the origin of the robot frame)
                # pointCloud -= np.array([camPos])
                #
                # # we further transform the point cloud to the frame at the robot_feet by adding the height h_t
                # pointCloud[:, 2] += camPos[2]
                #
                # # we then project the point cloud onto the grid world
                #
                # np.save("pcld", pointCloud)

                # bev_img = birds_eye_point_cloud(pointCloud, min_height=-2, max_height=2)

                axarr[0, 0].imshow(rgb)
                axarr[0, 1].imshow(realDepthImg)
                axarr[1, 0].imshow(seg)
                # axarr[1, 1].imshow(bev_img)
                plt.pause(0.01)

            # perception()

        print("Wallclock time: {}".format(time.time() - start_time))


def get_vision_observation_isaac(gym, sim, env, robot_handle, return_label=False):
    # 获取机器人在世界坐标系中的位置和旋转

    # rigid_body_states = gym.get_actor_rigid_body_states(env, robot_handle, gymapi.STATE_ALL)

    # 获取基座的位置信息和旋转信息（假设第一个刚体是基座）
    # base_state = rigid_body_states[0]  # 根据具体的机器人模型调整索引
    # position = base_state['pose']['p']  # 包含 x, y, z 位置
    # rotation = base_state['pose']['r']  # 包含 x, y, z, w 四元数旋转
    rot_pos = robot_handle.base_position[:, :].squeeze()[0],  # Base height
    rot_quat = robot_handle.base_orientation_quat[:, :].squeeze(),  # Base roll
    print(f"rot_quat: {rot_quat}")
    # transform = gym.get_actor_transform(env, robot_handle)
    # rot_pos = [transform.p.x, transform.p.y, transform.p.z]
    # rot_quat = [transform.r.x, transform.r.y, transform.r.z, transform.r.w]

    # 设置摄像机参数
    width = 240
    height = 240
    fov = 90
    near_val = 0.1
    far_val = 5

    # 计算投影矩阵
    aspect = width / height
    proj_mat = gymapi.CameraProperties()
    proj_mat.width = width
    proj_mat.height = height
    proj_mat.horizontal_fov = fov
    proj_mat.near_plane = near_val
    proj_mat.far_plane = far_val

    # 计算摄像机位置和方向
    def quaternion_to_rotation_matrix(quat):
        """
        将四元数转换为旋转矩阵
        """
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]

        # 计算旋转矩阵
        rotation_matrix = np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        ])
        return rotation_matrix

    rot_mat = quaternion_to_rotation_matrix(*rot_quat)
    forward_vec = np.dot(rot_mat, np.array([1, 0, 0]))
    cam_pos = [rot_pos[i] + forward_vec[i] * 0.239 for i in range(3)]
    cam_orn = rot_quat

    cam_up_vec = np.dot(rot_mat, np.array([0, 0, 1]))
    cam_target = [cam_pos[i] + forward_vec[i] * 10 for i in range(3)]

    # 创建摄像机
    camera_handle = gym.create_camera_sensor(env, proj_mat)
    gym.set_camera_transform(camera_handle, env, gymapi.Transform(
        p=gymapi.Vec3(*cam_pos),
        r=gymapi.Quat(*cam_orn)
    ))

    # 渲染摄像机图像
    gym.render_all_camera_sensors(sim)
    rgb_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    depth_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
    seg_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)

    rgb = torch.as_tensor(rgb_tensor).cpu().numpy()
    depth = torch.as_tensor(depth_tensor).cpu().numpy()
    seg = torch.as_tensor(seg_tensor).cpu().numpy()

    if return_label:
        info = {
            "cam_pos": cam_pos,
            "cam_orn": cam_orn,
            "rot_pos": rot_pos,
            "rot_quat": rot_quat,
            "width": width,
            "height": height
        }
    else:
        info = None

    return rgb, depth, seg, info


def get_vision_observation(gym, sim, env, camera_handle, i, return_label=False):
    # rot_pos, rot_orn = self.pybullet_client.getBasePositionAndOrientation(self._robot.quadruped)

    width = 240
    height = 240
    fov = 90
    near_val = 0.1
    far_val = 5

    aspect = width / height
    # proj_mat = self.pybullet_client.computeProjectionMatrixFOV(fov,
    #                                                            aspect,
    #                                                            near_val,
    #                                                            far_val)
    proj_mat = np.matrix(gym.get_camera_proj_matrix(sim, env, camera_handle))
    print(f"proj_mat: {proj_mat}")
    # rot_mat = self.pybullet_client.getMatrixFromQuaternion(rot_orn)
    # forward_vec = [rot_mat[0], rot_mat[3], rot_mat[6]]
    # cam_pos = [rot_pos[i] + forward_vec[i] * 0.239 for i in range(3)]
    # import copy
    # cam_orn = copy.deepcopy(rot_orn)
    # forward_vec2 = [rot_mat[0], rot_mat[3], rot_mat[6]]
    # cam_up_vec = [rot_mat[2], rot_mat[5], rot_mat[8]]

    # cam_target = [cam_pos[i] + forward_vec2[i] * 10 for i in range(3)]

    # view_mat2 = self.pybullet_client.computeViewMatrix(cam_pos, cam_target, cam_up_vec)
    view_mat = np.matrix(gym.get_camera_view_matrix(sim, env, camera_handle))

    # camera_image_set = self.pybullet_client.getCameraImage(
    #     width, height, viewMatrix=view_mat2, projectionMatrix=proj_mat,
    #     shadow=1,
    #     lightDirection=[1, 1, 1],
    #     renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
    #     flags=self.pybullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    # )

    camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    from isaacgym import gymtorch
    torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
    # save images
    img = torch_camera_tensor.permute(2, 0, 1)
    RGB = img[:3]  # This will give you the first 3 channels (RGB)
    alpha = img[3]  # This will give you the 4th channel (alpha)
    # Scale RGB channels to [0, 255]
    RGB = RGB * 255
    # Add an extra dimension to alpha so it can be concatenated with RGB
    alpha = alpha.unsqueeze(0)
    # Concatenate alpha and RGB channels
    img = torch.cat((RGB, alpha), dim=0)
    # Convert to PIL Image
    from torchvision.transforms.functional import to_pil_image
    img = to_pil_image(img)
    # Save image
    img.save(f'image{i}.png')

    rgb = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    depth = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_DEPTH)
    seg = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_SEGMENTATION)
    # imgW, imgH, rgb, depth, seg = camera_image_set
    imgW, imgH = 900, 1600
    if return_label:
        info = {
            # "cam_pos": cam_pos, "cam_orn": cam_orn,
            # "rot_pos": rot_pos, "rot_orn": rot_orn,
            "view_matrix": view_mat,
            "projection_matrix": proj_mat,
            "width": imgW,
            "height": imgH
        }
    else:
        info = None

    return rgb, depth, seg, info


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
