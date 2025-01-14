import time
from typing import List, Any

import os
import cv2
import numpy as np
import open3d as o3d
import skfmm
from matplotlib import pyplot as plt

from isaacgym import gymapi, gymtorch
from src.envs.robots.modules.planner.utils import to_ply
from src.envs.robots.modules.planner.utils import birds_eye_point_cloud
from src.envs.robots.modules.planner.FMM import FMMPlanner


class RGBDCamera:
    def __init__(self,
                 sim: Any,
                 env: Any,
                 viewer: Any,
                 attached_rigid_body_index_in_env,
                 resolution=(400, 400)):

        self._sim = sim
        self._gym = gymapi.acquire_gym()
        self._viewer = viewer
        self._env_handle = env
        self._img_width = resolution[0]
        self._img_height = resolution[1]
        self._attached_rigid_body = attached_rigid_body_index_in_env
        self._camera_handle = self.add_camera()

        self._video_cnt = 0

    def add_camera(self):
        # create camera
        camera_props = gymapi.CameraProperties()
        camera_props.width = self._img_width
        camera_props.height = self._img_height
        camera_props.enable_tensors = True  # Enable tensor output for the camera
        camera_props.near_plane = 0.1  # Minimum distance
        camera_props.far_plane = 10.0  # Maximum distance
        camera_horizontal_fov = 87
        camera_props.horizontal_fov = camera_horizontal_fov
        camera_handle = self._gym.create_camera_sensor(self._env_handle, camera_props)
        camera_pos = [0, 0, 2]

        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*camera_pos)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.radians(30.0))
        self._gym.attach_camera_to_body(camera_handle, self._env_handle, self._attached_rigid_body, local_transform,
                                        gymapi.FOLLOW_TRANSFORM)
        return camera_handle

    def get_vision_observation(self):
        width = self._img_width
        height = self._img_height
        # fov = 90
        # near_val = 0.1
        # far_val = 5

        proj_mat = self._gym.get_camera_proj_matrix(self._sim, self._env_handle, self._camera_handle)

        cam_transform = self._gym.get_camera_transform(self._sim, self._env_handle, self._camera_handle)
        cam_pos = cam_transform.p
        cam_orn = cam_transform.r

        view_mat2 = self._gym.get_camera_view_matrix(self._sim, self._env_handle, self._camera_handle)

        self._gym.render_all_camera_sensors(self._sim)
        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        color_image = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_COLOR)
        # depth_tensor = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle,
        #                                          gymapi.IMAGE_DEPTH)
        depth_image_ = self._gym.get_camera_image_gpu_tensor(self._sim,
                                                             self._env_handle,
                                                             self._camera_handle,
                                                             gymapi.IMAGE_DEPTH)

        torch_camera_depth_tensor = gymtorch.wrap_tensor(depth_image_)
        # Clamp depth values to the range [near_plane, far_plane]
        near_plane = 0.1
        far_plane = 10.0
        # torch_camera_depth_tensor = torch.clamp(torch_camera_depth_tensor, min=near_plane, max=far_plane)
        print(f"torch_camera_depth_tensor: {torch_camera_depth_tensor}")

        _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()

        # depth_image = gymtorch.wrap_tensor(depth_image_)
        # depth_image = self.process_depth_image(depth_image, i)

        self._gym.end_access_image_tensors(self._sim)

        # for i in range(self.num_envs):
        #     depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
        #                                                         self.envs[i],
        #                                                         self.cam_handles[i],
        #                                                         gymapi.IMAGE_DEPTH)
        #
        #     depth_image = gymtorch.wrap_tensor(depth_image_)
        # depth_image = self.process_depth_image(depth_image, i)

        # init_flag = self.episode_length_buf <= 1
        # if init_flag[i]:
        #     self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
        # else:
        #     self.depth_buffer[i] = torch.cat(
        #         [self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
        #         dim=0)

        # self._gym.end_access_image_tensors(self._sim)

        # depth_image = np.array(depth_image.cpu(), copy=True).reshape((1920, 1080))
        # position, velocity = get_ball_state(env, sphere_handle)
        # print("pos:{} vel{} ".format(position, velocity))

        rgba_image = np.frombuffer(color_image, dtype=np.uint8).reshape(self._img_height, self._img_width, 4)

        rgb_image = rgba_image[:, :, :3]
        # print(f"rgb_image: {rgb_image}")

        # time.sleep(1)
        # optical_flow_in_pixels = np.zeros(np.shape(optical_flow_image))
        # # Horizontal (u)
        # optical_flow_in_pixels[0, 0] = image_width * (optical_flow_image[0, 0] / 2 ** 15)
        # # Vertical (v)
        # optical_flow_in_pixels[0, 1] = image_height * (optical_flow_image[0, 1] / 2 ** 15)

        # self._frames.append(rgb_image)

        # rgb_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)
        # cv2.imshow('RGB Image', rgb_img)

        # _depth_img[np.isinf(_depth_img)] = -256

        # cv2.imshow('Depth Image', depth_colored)
        # cv2.waitKey(1)
        # print(f"_depth_img: {_depth_img}")
        # print(f"_depth_img: {_depth_img.shape}")
        #
        # print(f"depth_normalized: {depth_normalized}")
        # print(f"depth_normalized: {depth_normalized.shape}")
        # print(f"depth_colored: {depth_colored}")
        # print(f"depth_colored: {depth_colored.shape}")
        # is_all_zero = np.count_nonzero(depth_normalized) == 0

        # color_img = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_COLOR)
        # depth_img = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_DEPTH)

        seg_img = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle,
                                             gymapi.IMAGE_SEGMENTATION)

        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        color_image = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_COLOR)
        # depth_tensor = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle,
        #                                          gymapi.IMAGE_DEPTH)
        depth_image_ = self._gym.get_camera_image_gpu_tensor(self._sim,
                                                             self._env_handle,
                                                             self._camera_handle,
                                                             gymapi.IMAGE_DEPTH)

        torch_camera_depth_tensor = gymtorch.wrap_tensor(depth_image_)

        _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()

        # depth_image = gymtorch.wrap_tensor(depth_image_)
        # depth_image = self.process_depth_image(depth_image, i)

        self._gym.end_access_image_tensors(self._sim)

        color_img = color_image
        rgba_img = np.frombuffer(color_img, dtype=np.uint8).reshape(self._img_height, self._img_width, 4)

        # rgb_img = rgba_img[:, :, :3]
        # print(f"rgb_image: {rgb_image}")

        # time.sleep(1)
        # optical_flow_in_pixels = np.zeros(np.shape(optical_flow_image))
        # # Horizontal (u)
        # optical_flow_in_pixels[0, 0] = image_width * (optical_flow_image[0, 0] / 2 ** 15)
        # # Vertical (v)
        # optical_flow_in_pixels[0, 1] = image_height * (optical_flow_image[0, 1] / 2 ** 15)

        depth_normalized = cv2.normalize(_depth_img, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        np.savetxt("depth_normalized.txt", depth_normalized, fmt="%.6f")
        print(f"rgb_img: {rgba_img}")
        print(f"rgb_img: {rgba_img.shape}")
        print(f"depth: {_depth_img}")
        print(f"depth: {_depth_img.shape}")
        print(f"depth normalized: {depth_normalized}")
        print(f"depth normalized: {depth_normalized.shape}")
        # time.sleep(123)
        return rgba_img, depth_normalized, seg_img

    def get_current_frame(self):
        o3d_pcd = self.get_pcd_data()

        # bev_img = birds_eye_point_cloud(
        #     points=np.asarray(o3d_pcd.points),
        #     side_range=(-1, 107),
        #     fwd_range=(-110, 114)
        # )
        bev_img = self.get_bev_map(
            raw_pcd=np.asarray(o3d_pcd.points),
            as_occupancy=False,
        )
        # occupancy_map = self.get_occupancy_map(bev_map=bev_img)
        occupancy_map = bev_img
        # occupancy_map = np.where(bev_img == 0, 1, 0)
        np.savetxt(f"occ.txt", occupancy_map, fmt="%d")

        goal = [500, 300]
        costmap = self.get_costmap(occupancy_map, goal)

        # Plot
        # plt.subplot(1, 2, 1)
        # plt.imshow(occupancy_map, origin='lower')
        # plt.colorbar(label="Occupancy")
        # plt.title("Occupancy Map")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(costmap, cmap='viridis', origin='lower')
        # # plt.imshow(costmap, cmap='jet', origin='lower')
        # # plt.scatter(*start[::-1], c='blue', label='Start')
        # plt.scatter(*goal[::-1], c='red', s=20, label='Goal')
        # plt.title("Cost Map")
        # plt.colorbar(label="Cost", cmap='jet')
        # plt.legend()
        # plt.show()

        # print("loaded!!!!")
        # time.sleep(123)
        # if show_plot:
        #     axarr[0, 0].imshow(rgb)
        #     axarr[0, 1].imshow(realDepthImg)
        #     axarr[1, 0].imshow(seg)
        #     axarr[1, 1].imshow(bev_img)
        #     plt.pause(0.1)

        label_save_folder = '.'
        bev_save_path = os.path.join(label_save_folder, f"bev_.png")
        plt.imsave(bev_save_path, bev_img)

        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        color_image = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle,
                                                 gymapi.IMAGE_COLOR)
        print(f"color_image: {color_image}")
        print(f"color_image: {type(color_image)}")
        print(f"color_image: {color_image.shape}")

        # depth_tensor = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle,
        #                                          gymapi.IMAGE_DEPTH)
        depth_image_ = self._gym.get_camera_image_gpu_tensor(self._sim, self._env_handle, self._camera_handle,
                                                             gymapi.IMAGE_DEPTH)

        torch_camera_depth_tensor = gymtorch.wrap_tensor(depth_image_)
        # Clamp depth values to the range [near_plane, far_plane]
        near_plane = 0.1
        far_plane = 10.0
        # torch_camera_depth_tensor = torch.clamp(torch_camera_depth_tensor, min=near_plane, max=far_plane)
        print(f"torch_camera_depth_tensor: {torch_camera_depth_tensor}")

        _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()

        # depth_image = gymtorch.wrap_tensor(depth_image_)
        # depth_image = self.process_depth_image(depth_image, i)

        self._gym.end_access_image_tensors(self._sim)

        # for i in range(self.num_envs):
        #     depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim,
        #                                                         self.envs[i],
        #                                                         self.cam_handles[i],
        #                                                         gymapi.IMAGE_DEPTH)
        #
        #     depth_image = gymtorch.wrap_tensor(depth_image_)
        # depth_image = self.process_depth_image(depth_image, i)

        # init_flag = self.episode_length_buf <= 1
        # if init_flag[i]:
        #     self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
        # else:
        #     self.depth_buffer[i] = torch.cat(
        #         [self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)],
        #         dim=0)

        # self._gym.end_access_image_tensors(self._sim)

        # depth_image = np.array(depth_image.cpu(), copy=True).reshape((1920, 1080))
        # position, velocity = get_ball_state(env, sphere_handle)
        # print("pos:{} vel{} ".format(position, velocity))

        rgba_image = np.frombuffer(color_image, dtype=np.uint8).reshape(self._img_height, self._img_width, 4)

        rgb_image = rgba_image[:, :, :3]

        # print(f"rgb_image: {rgb_image}")

        # time.sleep(1)
        # optical_flow_in_pixels = np.zeros(np.shape(optical_flow_image))
        # # Horizontal (u)
        # optical_flow_in_pixels[0, 0] = image_width * (optical_flow_image[0, 0] / 2 ** 15)
        # # Vertical (v)
        # optical_flow_in_pixels[0, 1] = image_height * (optical_flow_image[0, 1] / 2 ** 15)

        # self._frames.append(rgb_image)

        rgb_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        # # _depth_img[np.isinf(_depth_img)] = -256
        depth_normalized = cv2.normalize(_depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        # # cv2.imshow('Depth Image', depth_normalized)
        # # cv2.imshow('Depth Color Image', depth_colored)
        cv2.imwrite(f'rgb_{self._video_cnt}.png', rgb_img)
        cv2.imwrite(f'depth_color_{self._video_cnt}.png', depth_colored)
        cv2.imwrite(f'depth_{self._video_cnt}.png', depth_normalized)
        cv2.waitKey(1)

        cv2.imshow('RGB Image', rgb_img)

        # _depth_img[np.isinf(_depth_img)] = -256

        _depth_img = self.replace_inf_with_second_smallest(_depth_img)

        depth_normalized = cv2.normalize(_depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colored = depth_normalized
        cv2.imshow('Depth Image', depth_colored)

        # combined_image = np.hstack((rgb_img, depth_colored))
        # cv2.imshow('Image', combined_image)
        cv2.waitKey(1)
        print(f"_depth_img: {_depth_img}")
        print(f"_depth_img: {_depth_img.shape}")

        print(f"depth_normalized: {depth_normalized}")
        print(f"depth_normalized: {depth_normalized.shape}")
        print(f"depth_colored: {depth_colored}")
        print(f"depth_colored: {depth_colored.shape}")
        is_all_zero = np.count_nonzero(depth_normalized) == 0
        if is_all_zero:
            np.savetxt('depth_error.txt', _depth_img)
            # np.savetxt('raw_depth_tensor.txt', depth_image_)
            # torch.save(torch_camera_depth_tensor, 'raw_depth_tensor_error.pt')
            # time.sleep(123)
        else:
            # np.savetxt('depth.txt', _depth_img)
            # torch.save(torch_camera_depth_tensor, 'raw_depth_tensor.pt')
            pass
        return _depth_img

    def replace_inf_with_second_smallest(self, depth_image):
        """
        Replace all `inf` values in a depth image with the second-smallest finite value.

        Args:
            depth_image (np.ndarray): Input depth image (2D array).

        Returns:
            np.ndarray: Depth image with `inf` values replaced.
        """
        print(f"depth_image: {depth_image}")

        # Flatten the array and filter finite values
        finite_values = depth_image[np.isfinite(depth_image)]

        if len(finite_values) < 2:
            raise ValueError(
                "The depth image does not have enough finite values to determine the second smallest.")

        # Find the unique sorted finite values
        unique_values = np.unique(finite_values)

        if len(unique_values) < 2:
            raise ValueError(
                "The depth image does not have enough unique finite values for a valid replacement.")

        # The second smallest value
        second_smallest = unique_values[1]

        # Replace `inf` values with the second-smallest value
        result = np.copy(depth_image)
        result[np.isinf(result)] = second_smallest

        return result

    def get_bev_map(self,
                    raw_pcd: np.ndarray,
                    x_range=(45, 55),
                    y_range=(-5, 5),
                    z_range=(-20, 20),
                    grid_size=0.015,
                    as_occupancy=False,
                    reverse_xy=True,
                    show_bev=False):
        """ Creates an 2D birds eye view representation of the point cloud data.

            Args:
                raw_pcd:     (numpy array)
                            N rows of raw point cloud data
                            Each point should be specified by at least 3 elements x,y,z
                x_range:    (tuple of two floats)
                            (left, right) in metres
                            left and right limits of rectangle to look at.
                y_range:    (tuple of two floats)
                            (behind, front) in metres
                            back and front limits of rectangle to look at.
                grid_size:  (float) desired resolution in metres to use
                            Each output pixel will represent a square region res x res
                            in size.
                z_range:    (tuple of two floats)
                            (low, high) in metres
                            low and high limits of rectangle to look at.
                as_occupancy: (boolean)(default=False)
                              To generate an occupancy map or not
                reverse_xy: reverse the bev image of x and y-axis or not
            """

        # Region of Interest (ROI)
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range  # Optional filtering by height

        # Filter points within the ROI
        roi_points = raw_pcd[
            (raw_pcd[:, 0] >= x_min) & (raw_pcd[:, 0] <= x_max) &
            (raw_pcd[:, 1] >= y_min) & (raw_pcd[:, 1] <= y_max) &
            (raw_pcd[:, 2] >= z_min) & (raw_pcd[:, 2] <= z_max)]

        # Step 3: Define grid parameters for BEV map
        x_bins = int((x_max - x_min) / grid_size)
        y_bins = int((y_max - y_min) / grid_size)

        # Initialize BEV map (height map)
        bev_map = np.zeros((x_bins, y_bins), dtype=np.float32)

        # if as_occupancy:
        #     # initialize as unknown
        #     # mask unknown as -1
        #     # occupied as 1
        #     # free as 0
        #     im = -1 * np.ones([y_max, x_max], dtype=np.uint8)  # initialize grid as unknown (-1)
        #     height = z_lidar[indices]
        #     height[height > min_height] = 1
        #     height[height <= min_height] = 0
        #     pixel_values = scale_to_255(height, min=-1, max=1)
        #     im[-y_img, x_img] = pixel_values

        # Populate BEV map with max height
        for point in roi_points:
            x, y, z = point
            # print(f"{idx}: {x}, {y}, {z}")
            x_idx = int((x - x_min) / grid_size) - 1
            y_idx = int((y - y_min) / grid_size) - 1
            # if as_occupancy:
            #     # z = 1 if z > z_min else 0
            #     z = 1 if z > 0.06 else 0
            #     bev_map[x_idx, y_idx] = z  # Use max height for the grid cell
            # else:
            bev_map[x_idx, y_idx] = max(bev_map[x_idx, y_idx], z)  # Use max height for the grid cell

        if as_occupancy:
            z_low, z_high = -0.1, 0.2

            occupancy_grid = np.ones([x_bins, y_bins], dtype=np.uint8) * np.nan

            for point in roi_points:
                x, y, z = point
                # print(f"{idx}: {x}, {y}, {z}")
                x_idx = int((x - x_min) / grid_size) - 1
                y_idx = int((y - y_min) / grid_size) - 1
                occupancy_grid[x_idx, y_idx] = z  # Use max height for the grid cell

            # occupancy_grid[-y_img, x_img] = z_height

            x, y = np.indices(occupancy_grid.shape)

            interp_grid = np.array(occupancy_grid)
            from scipy.interpolate import griddata
            interp_grid[np.isnan(interp_grid)] = griddata((x[~np.isnan(interp_grid)], y[~np.isnan(interp_grid)]),
                                                          # points we know
                                                          interp_grid[~np.isnan(interp_grid)],
                                                          (x[np.isnan(interp_grid)], y[np.isnan(interp_grid)]))
            z_height_feasible_mask = np.logical_and((interp_grid.copy() > z_low), (interp_grid.copy() < z_high))
            occupancy_map = np.ones([x_bins, y_bins], dtype=np.uint8) * z_height_feasible_mask

            # Original Code
            # occupancy_map = -1 * np.ones((x_bins, y_bins), dtype=np.uint8)
            # for point in roi_points:
            #     x, y, z = point
            #     # print(f"{idx}: {x}, {y}, {z}")
            #     x_idx = int((x - x_min) / grid_size) - 1
            #     y_idx = int((y - y_min) / grid_size) - 1
            #
            #     if z_low < z < z_high:
            #         height = 0
            #     else:
            #         height = 1
            #     occupancy_map[x_idx, y_idx] = height

            # z_height_feasible_mask = np.logical_and((bev_map[:, 2] > z_low), (bev_map[:, 2] < z_high))
            # print(f"z_height_feasible_mask: {z_height_feasible_mask}")
            # print(f"z_height_feasible_mask: {z_height_feasible_mask.shape}")
            # occupancy_map = np.ones((x_bins, y_bins), dtype=np.uint8) * z_height_feasible_mask
            # occupancy_map[occupancy_map == -1] = 0
            bev_map = occupancy_map

        # bev_map = 1 - bev_map

        # Planning
        # planner = FMMPlanner(bev_map)
        # goal = [47, 1]
        # goal_bev_x = int((goal[0] - x_min) / grid_size) - 1
        # goal_bev_y = int((goal[0] - x_min) / grid_size) - 1
        # goal_in_bev = [goal_bev_x, goal_bev_y]
        #
        # dd = planner.set_goal(goal)
        #
        # def scale_to_255(a, min, max, dtype=np.uint8):
        #     """ Scales an array of values from specified min, max range to 0-255
        #         Optionally specify the data type of the output (default is uint8)
        #     """
        #     return (((a - min) / float(max - min)) * 255).astype(dtype)
        #
        # np.savetxt("distance.txt", dd, fmt="%.2f")
        # distance_map = scale_to_255(np.array(dd), min=0, max=80)
        #
        # dd = skfmm.distance(phi=np.where(bev_map == 1, 0, 1))  # 障碍物为 0，其他区域为正值)
        # distance_map = dd

        if show_bev:
            # Visualize BEV Map
            plt.figure(figsize=(10, 8))
            if reverse_xy:
                plt.imshow(bev_map, cmap='viridis', origin='lower', extent=[y_min, y_max, x_min, x_max])
                plt.xlabel('Y (m)')
                plt.ylabel('X (m)')
            else:
                plt.imshow(bev_map.T, cmap='viridis', origin='lower', extent=[x_min, x_max, y_min, y_max])
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
            # plt.imshow(distance_map.T)
            plt.title('BEV Map (Height Map)')
            plt.colorbar(label='Height (m)')
            plt.show()

        return bev_map

    def get_occupancy_map(self,
                          bev_map: np.ndarray,
                          height_range=(-0.1, 0.2)):

        z_low, z_high = height_range
        occupancy_map = -1 * np.ones_like(bev_map, dtype=np.uint8)

        for i in range(bev_map.shape[0]):
            for j in range(bev_map.shape[1]):
                z = bev_map[i, j]

                if z_low < z < z_high:
                    height = 0
                else:
                    height = 1
                occupancy_map[i, j] = height

        return occupancy_map

    def get_costmap(self, occupancy_map, goal):
        """Return the generated costmap from occupancy_map and goal.

        Parameters
        ----------
        occupancy_map : array-like
                        occupancy_map with element 0 being the obstacles
                        and element 1 being the free spaces

        goal : 2d array/list
               target position on the map with goal = (target_x, target_y)
        """
        _planner = FMMPlanner(occupancy_map)
        dd = _planner.set_goal(goal)
        print(f"dd: {dd}")
        np.savetxt(f"dd.txt", dd, fmt="%.2f")
        return dd

    def get_pcd_data(self, in_world_frame=True) -> o3d.geometry.PointCloud:
        """
        Obtain the point cloud data
        :param in_world_frame:
        """
        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        # Depth buffer
        _depth = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_DEPTH)

        # Segmentation buffer
        _seg = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_SEGMENTATION)

        # Point Cloud w.r.t world frame or camera frame
        if in_world_frame:
            # in world frame
            view_mat = self._gym.get_camera_view_matrix(self._sim, self._env_handle, self._camera_handle)
        else:
            view_mat = np.identity(4)  # In camera frame

        # Inverse of View Matrix
        view_mat_inv = np.linalg.inv(np.matrix(view_mat))

        # Camera Projection Matrix
        proj_mat = self._gym.get_camera_proj_matrix(self._sim, self._env_handle, self._camera_handle)

        imgW = self._img_width
        imgH = self._img_height

        # Get the camera projection matrix and get the necessary scaling
        # coefficients for de-projection
        fu = 2 / proj_mat[0, 0]
        fv = 2 / proj_mat[1, 1]

        MIN_THRESH = -10001
        # Ignore any points which originate from ground plane or empty space
        # _depth[_seg == 0] = MIN_THRESH

        print("Converting Depth images to point clouds. Have patience...")
        points = []
        cam_width = imgW
        cam_height = imgH
        u0 = cam_width / 2
        v0 = cam_height / 2
        for u in range(cam_width):
            for v in range(cam_height):

                # Ignore abnormal depth data
                if _depth[v, u] <= MIN_THRESH:
                    continue

                # Obtain regions with item (0 represents null)
                # if _seg[v, u] > 0:
                u_term = (u - u0) / cam_width  # image-space coordinate (u)
                v_term = (v - v0) / cam_height  # image-space coordinate (v)
                d = _depth[v, u]  # depth.shape = (width, height)
                Pc = [d * fu * u_term, d * fv * v_term, d, 1]  # Pos in camera frame
                Pw = Pc * view_mat_inv  # Pos in world frame
                points.append([Pw[0, 0], Pw[0, 1], Pw[0, 2]])  # [x, y, z]

        # Get RGB image
        _rgb_buffer = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_COLOR)
        rgba_image = np.frombuffer(_rgb_buffer, dtype=np.uint8).reshape(self._img_height, self._img_width, 4)
        rgb_image = rgba_image[:, :, :3]

        points = np.asarray(points)
        colors = rgb_image.reshape(-1, *rgb_image.shape[2:])

        # Compose point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Visualize the point cloud
        # o3d.visualization.draw_geometries([point_cloud],
        #                                   zoom=0.8,
        #                                   front=[0.0, 0.0, -1.0],
        #                                   lookat=[0.0, 0.0, 0.0],
        #                                   up=[0.0, -1.0, 0.0])

        # Save to ply file
        filename = "pcd.ply"
        o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)
        print(f"Point cloud data saved to {filename}")

        self._gym.end_access_image_tensors(self._sim)

        return point_cloud

    def save_img(self, img_data, filename, folder="./data/perception"):
        os.makedirs(folder, exist_ok=True)
        save_path = os.path.join(folder, filename)
        plt.imsave(save_path, img_data)
        print(f"image successfully saved to path: {save_path}")

    @property
    def camera_handle(self):
        return self._camera_handle

    @property
    def resolution(self) -> List:
        return [self._img_width, self._img_height]
