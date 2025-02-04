import time

import os
import cv2
import numpy as np
import open3d as o3d
from typing import List, Any

import pylab as p
from matplotlib import pyplot as plt
from isaacgym import gymapi, gymtorch
from scipy.interpolate import griddata
from scipy.spatial.transform import Rotation


# class BEVMap:
#     def __init__(self,
#                  roi_x_range=(45, 55),
#                  roi_y_range=(-5, 5),
#                  roi_z_range=(-20, 20),
#                  resolution=0.015):
#         self.roi_x_range = roi_x_range  # ROI (x range)
#         self.roi_y_range = roi_y_range  # ROI (y range)
#         self.roi_z_range = roi_z_range  # ROI (z range)
#         self.resolution = resolution  # Resolution


class RGBDCamera:
    def __init__(self,
                 robot: Any,
                 sim: Any,
                 env: Any,
                 viewer: Any,
                 attached_rigid_body_index_in_env,
                 resolution=(640, 480)):

        self._robot = robot
        self._sim = sim
        self._gym = gymapi.acquire_gym()
        self._viewer = viewer
        self._env_handle = env
        self._img_width = resolution[0]
        self._img_height = resolution[1]
        self._attached_rigid_body = attached_rigid_body_index_in_env
        self._camera_handle = self.add_camera()

        self._video_cnt = 0
        self._frame_cnt = 0

        self._camera_gpu_warmup()

    def _camera_gpu_warmup(self):
        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        color_image = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_COLOR)
        _depth_img = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_DEPTH)

        self._gym.end_access_image_tensors(self._sim)

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

    def get_current_frame(self, img_save=True, img_show=True):

        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        color_image = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_COLOR)
        _depth_img = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_DEPTH)

        rgba_image = np.frombuffer(color_image, dtype=np.uint8).reshape(self._img_height, self._img_width, 4)
        rgb_image = rgba_image[:, :, :3]

        # depth_image_ = self._gym.get_camera_image_gpu_tensor(self._sim, self._env_handle, self._camera_handle,
        #                                                      gymapi.IMAGE_DEPTH)
        # torch_camera_depth_tensor = gymtorch.wrap_tensor(depth_image_)
        # _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()

        self._gym.end_access_image_tensors(self._sim)

        # optical_flow_in_pixels = np.zeros(np.shape(optical_flow_image))
        # # Horizontal (u)
        # optical_flow_in_pixels[0, 0] = image_width * (optical_flow_image[0, 0] / 2 ** 15)
        # # Vertical (v)
        # optical_flow_in_pixels[0, 1] = image_height * (optical_flow_image[0, 1] / 2 ** 15)

        rgb_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        # # _depth_img[np.isinf(_depth_img)] = -256
        depth_normalized = cv2.normalize(_depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        # # cv2.imshow('Depth Image', depth_normalized)
        # # cv2.imshow('Depth Color Image', depth_colored)
        cv2.imwrite(f'rgb_{self._video_cnt}.png', rgb_img)
        cv2.imwrite(f'depth_color_{self._video_cnt}.png', depth_colored)
        cv2.imwrite(f'depth_{self._video_cnt}.png', depth_normalized)

        np.savetxt(f"depth_img.txt", _depth_img, fmt="%.2f")

        _depth_img = self.replace_inf_with_second_smallest(_depth_img)

        depth_normalized = cv2.normalize(_depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colored = depth_normalized

        # Show image flag
        if img_show:
            cv2.imshow('RGB Image', rgb_img)
            cv2.imshow('Depth Image', depth_colored)

        # combined_image = np.hstack((rgb_img, depth_colored))
        # cv2.imshow('Image', combined_image)
        cv2.waitKey(1)
        # print(f"_depth_img: {_depth_img}")
        # print(f"_depth_img: {_depth_img.shape}")

        # print(f"depth_normalized: {depth_normalized}")
        # print(f"depth_normalized: {depth_normalized.shape}")
        # print(f"depth_colored: {depth_colored}")
        # print(f"depth_colored: {depth_colored.shape}")
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

        self._frame_cnt += 1
        return _depth_img

    def replace_inf_with_second_smallest(self, depth_image):
        """
        Replace all `inf` values in a depth image with the second-smallest finite value.

        Args:
            depth_image (np.ndarray): Input depth image (2D array).

        Returns:
            np.ndarray: Depth image with `inf` values replaced.
        """
        # print(f"depth_image: {depth_image}")

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
                    roi_x_range=(45, 55),
                    roi_y_range=(-5, 5),
                    roi_z_range=(-20, 20),
                    grid_size=0.015,
                    reverse_xy=True,
                    show_map=False,
                    as_occupancy=False,
                    occupancy_z_range=(-0.1, 0.2),
                    save_map=False,
                    ):
        """ Creates an 2D birds eye view representation of the point cloud data.

            Args:
                roi_x_range: (tuple of two floats) by default (45, 55)
                             (left, right) in metres
                             left and right limits of rectangle to look at.
                roi_y_range: (tuple of two floats) by default (-5, 5)
                             (behind, front) in metres
                             back and front limits of rectangle to look at.
                roi_z_range: (tuple of two floats) by default (-20, 20)
                             (low, high) in metres
                             low and high limits of rectangle to look at.
                grid_size:   (float) desired resolution in metres to use
                             Each output pixel will represent a square region res x res
                             in size.
                as_occupancy: (boolean)(default=False)
                              To generate an occupancy map or not (0 - obstacle, 1 - free space)
                reverse_xy: reverse the bev image of x and y-axis or not
                show_map:   whether to show the generated map
                occupancy_z_range: height range of occupancy map
                save_map:   whether to save the generated map
            """
        # Get point cloud data
        o3d_pcd_wd = self.get_pcd_data(in_world_frame=True, write_ply=False, filename="wd.ply")  # pcd in world frame
        raw_pcd_wd = np.asarray(o3d_pcd_wd.points)

        # camera_tf = self._gym.get_camera_transform(self._sim, self._env_handle, self._camera_handle)
        # camera_pose = camera_tf.p
        # camera_quat = np.array([camera_tf.r.x, camera_tf.r.y, camera_tf.r.z, camera_tf.r.w])
        # rot_matrix = Rotation.from_quat(camera_quat).as_matrix()
        # print(f"camera_pose: {camera_pose}")
        # print(f"camera_quat: {camera_quat}")
        # print(f"rot_matrix: {rot_matrix}")

        # pcd_cam = transform_pcd_world_to_camera(raw_pcd_wd, camera_transform=camera_tf)

        # Region of Interest (ROI) Center
        # roi_x_center = camera_pose.x
        # roi_y_center = camera_pose.y
        # roi_z_center = camera_pose.z

        # Region of Interest (ROI) Range
        x_min, x_max = roi_x_range
        y_min, y_max = roi_y_range
        z_min, z_max = roi_z_range  # Optional filtering by height

        # Filter points within the ROI
        raw_pcd = raw_pcd_wd
        raw_pcd[:, 1] = - raw_pcd[:, 1]  # The y-axis data of point cloud is in opposite direction!!!
        roi_points = raw_pcd[
            (raw_pcd[:, 0] >= x_min) & (raw_pcd[:, 0] <= x_max) &
            (raw_pcd[:, 1] >= y_min) & (raw_pcd[:, 1] <= y_max) &
            (raw_pcd[:, 2] >= z_min) & (raw_pcd[:, 2] <= z_max)
            ]

        # Step 3: Define grid parameters for BEV map
        x_bins = int((x_max - x_min) / grid_size)
        y_bins = int((y_max - y_min) / grid_size)

        # Initialize BEV map (height map)
        bev_map = np.zeros((x_bins, y_bins), dtype=np.float32)
        # bev_map = np.ones((x_bins, y_bins), dtype=np.float32)

        # Populate BEV map with max height
        for point in roi_points:
            x, y, z = point
            x_idx = int((x - x_min) / grid_size) - 1
            y_idx = int((y - y_min) / grid_size) - 1
            # print(f"x_idx, y_idx: {(x_idx, y_idx)}")
            bev_map[x_idx, y_idx] = max(bev_map[x_idx, y_idx], z)  # Use max height for the grid cell

        # Generate Occupancy Map
        if as_occupancy:
            z_low, z_high = occupancy_z_range
            occupancy_grid = np.ones([x_bins, y_bins], dtype=np.uint8) * np.nan

            for point in roi_points:
                x, y, z = point
                x_idx = int((x - x_min) / grid_size) - 1
                y_idx = int((y - y_min) / grid_size) - 1
                occupancy_grid[x_idx, y_idx] = z  # Use max height for the grid cell

            x, y = np.indices(occupancy_grid.shape)
            interp_grid = np.array(occupancy_grid)
            interp_grid[np.isnan(interp_grid)] = griddata((x[~np.isnan(interp_grid)], y[~np.isnan(interp_grid)]),
                                                          # points we know
                                                          interp_grid[~np.isnan(interp_grid)],
                                                          (x[np.isnan(interp_grid)], y[np.isnan(interp_grid)]))
            z_height_feasible_mask = np.logical_and((interp_grid.copy() > z_low), (interp_grid.copy() < z_high))
            occupancy_map = np.ones([x_bins, y_bins], dtype=np.uint8) * z_height_feasible_mask

            # occupancy_map[0:125, 200:220] = 1      # Manually fill the free-space for robot (optional)
            occupancy_map[0:140, 200:465] = 1  # Manually fill the free-space for robot (optional) resolution = 0.015
            # occupancy_map[0:70, 100:232] = 1  # Manually fill the free-space for robot (optional) resolution = 0.03

            bev_map = occupancy_map

        # Flip the BEV Map
        # bev_map = bev_map[::-1]
        # bev_map = np.fliplr(bev_map)
        # np.savetxt(f"bevmap.txt", bev_map, fmt="%.1f")

        # Store BEV Map
        if save_map:
            label_save_folder = '.'
            bev_save_path = os.path.join(label_save_folder, f"bev.png")
            plt.imsave(bev_save_path, bev_map)
            bev_save_path = os.path.join(label_save_folder, f"bev_flip_ud.png")
            plt.imsave(bev_save_path, np.flipud(bev_map))

        # Visualize BEV Map
        if show_map:
            plt.figure(figsize=(10, 8))
            if reverse_xy:
                plt.imshow(bev_map, cmap='viridis', origin='lower', extent=[y_min, y_max, x_min, x_max])
                # plt.imshow(bev_map, cmap='viridis')
                plt.xlabel('Y (m)')
                plt.ylabel('X (m)')
            else:
                plt.imshow(bev_map.T, cmap='viridis', origin='lower', extent=[x_min, x_max, y_min, y_max])
                # plt.imshow(bev_map, cmap='viridis')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
            map_name = "Occupancy" if as_occupancy else "BEV"
            plt.title(f'{map_name} Map (Height Map)')
            plt.colorbar(label='Height (m)')
            plt.show()

        return bev_map

    def get_depth_origin_world_frame(self, min_threshold=-10001):
        """Obtain figure origin pose from depth image in the world frame

        Below are how the u and v look like in a saved depth image (from human eyes)
        (u) ← ------------
              |          |
              |          |
               --------- |
                         ↓  (v)
        """
        self._gym.fetch_results(self._sim, True)
        self._gym.step_graphics(self._sim)
        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        # Depth buffer
        _depth = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_DEPTH)

        # View Matrix
        view_mat = self._gym.get_camera_view_matrix(self._sim, self._env_handle, self._camera_handle)

        # Inverse of View Matrix
        view_mat_inv = np.linalg.inv(np.matrix(view_mat))

        # Camera Projection Matrix
        proj_mat = self._gym.get_camera_proj_matrix(self._sim, self._env_handle, self._camera_handle)

        fu = 2 / proj_mat[0, 0]
        fv = 2 / proj_mat[1, 1]

        # Depth of the image origin (lower-left corner in depth image)
        u = self._img_width - 1
        v = self._img_height - 1

        # u = int(self._img_width / 5 * 4+40)
        # v = int(self._img_height / 2)

        d = _depth[v, u]
        if d <= min_threshold:
            # raise RuntimeError(f"error with the depth camera for the pixel-level origin")
            return [-np.inf, -np.inf, -np.inf], False

        cam_width = self._img_width
        cam_height = self._img_height
        u0 = cam_width / 2
        v0 = cam_height / 2

        u_term = (u - u0) / cam_width  # image-space coordinate (u)
        v_term = (v - v0) / cam_height  # image-space coordinate (v)
        Pc = [d * fu * u_term, d * fv * v_term, d, 1]  # Pos in camera frame
        Pw = Pc * view_mat_inv  # Pos in world frame

        # Get the pose of origin in world frame
        origin_in_world_frame = [Pw[0, 0], Pw[0, 1], Pw[0, 2]]

        return origin_in_world_frame, True

    def get_pcd_data(self, in_world_frame=True, visualize=False,
                     write_ply=False, filename="pcd.ply") -> o3d.geometry.PointCloud:
        """
        Obtain the point cloud data
        :param in_world_frame: In world frame or camera frame
        :param visualize: Flag to visualize the point cloud data
        :param write_ply: Write to ply file or not
        :param filename: filename and format to save the point cloud
        """
        self._gym.render_all_camera_sensors(self._sim)
        self._gym.start_access_image_tensors(self._sim)

        # Depth buffer
        _depth = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_DEPTH)
        # near_plane = 0.1
        # far_plane = 10.0
        # _depth = np.clip(_depth, a_min=near_plane, a_max=far_plane)

        # Segmentation buffer
        _seg = self._gym.get_camera_image(self._sim, self._env_handle, self._camera_handle, gymapi.IMAGE_SEGMENTATION)

        # View Matrix
        view_mat = self._gym.get_camera_view_matrix(self._sim, self._env_handle, self._camera_handle)

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
                Pc = [d * fu * u_term, d * fv * v_term, d, 1]  # Pos in camera view

                # Pos in world frame
                if in_world_frame:
                    Pw = Pc * view_mat_inv

                # Pos in camera frame
                else:
                    camera_tf = self._gym.get_camera_transform(self._sim, self._env_handle, self._camera_handle)
                    p_cam = np.array([camera_tf.p.x, camera_tf.p.y, camera_tf.p.z])
                    quat = np.array([camera_tf.r.x, camera_tf.r.y, camera_tf.r.z, camera_tf.r.w])
                    R_cam = Rotation.from_quat(quat).as_matrix()
                    HT_cam = np.eye(4)
                    HT_cam[:3, :3] = R_cam
                    HT_cam[:3, 3] = p_cam
                    Pw = Pc * view_mat_inv * np.linalg.inv(HT_cam)  # Pos in camera frame

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

        # Whether to save as ply file
        if write_ply:
            o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)
            print(f"Point cloud data saved to {filename}")

        # Visualize the point cloud
        if visualize:
            o3d.visualization.draw_geometries([point_cloud],
                                              zoom=0.8,
                                              front=[0.0, 0.0, -1.0],
                                              lookat=[0.0, 0.0, 0.0],
                                              up=[0.0, -1.0, 0.0])

        self._gym.end_access_image_tensors(self._sim)

        return point_cloud

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
        # print(f"torch_camera_depth_tensor: {torch_camera_depth_tensor}")

        _depth_img = torch_camera_depth_tensor.clone().cpu().numpy()

        self._gym.end_access_image_tensors(self._sim)

        rgba_image = np.frombuffer(color_image, dtype=np.uint8).reshape(self._img_height, self._img_width, 4)

        rgb_image = rgba_image[:, :, :3]
        # print(f"rgb_image: {rgb_image}")

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

        # time.sleep(123)
        return rgba_img, depth_normalized, seg_img

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
