import os
import time

import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib import utils
from matplotlib import pyplot as plt
from src.envs.robots.modules.planner.FMM import FMMPlanner
from src.envs.robots.modules.planner.spline import Spline


class PathPlanner:
    def __init__(self, robot, device="cuda"):
        self._device = device
        self._robot = robot
        self._goal = [500, 300]
        self._resolution = 0.015
        self._planner = None
        self._spine = Spline()
        self._origin_in_world = np.array([0., 0., 0.])

    # def update(self, occupancy_map):
    #     self._planner = FMMPlanner(occupancy_map, self._goal)
    #     self._planner.set_goal(self._goal)

    def project_to_bev_frame(self,
                             pose_world_frame,
                             grid_size=None,
                             roi_x_range=(45, 55),
                             roi_y_range=(-5, 5)):
        """Project the pose in world frame to the gridmap

        Below are how the x and y look like in a generated raw BEV image
        ------------> (y)
        |
        |
        |
        ↓  (x)
        """
        if grid_size is None:
            grid_size = self._resolution

        pose_xy_in_world = pose_world_frame[:2]
        x, y = pose_xy_in_world

        # Region of Interest (ROI)
        x_min, x_max = roi_x_range
        y_min, y_max = roi_y_range

        x_idx = int((x - x_min) / grid_size) - 1
        y_idx = int((y - y_min) / grid_size) - 1

        x_bins = int((x_max - x_min) / grid_size)
        y_bins = int((y_max - y_min) / grid_size)

        map_frame = [x_idx, y_idx]
        map_shape = [x_bins, y_bins]

        return map_frame, map_shape

    def world_to_map_frame(self, pose_in_world, env_idx=0, bev_frame=True):
        """Project the pose from world frame to BEV map frame

        The output is human-readable
        """
        origin_in_world, flag = self._robot.camera_sensor[env_idx].get_depth_origin_world_frame()
        if flag is True:
            self._origin_in_world = origin_in_world

        origin_in_map, map_shape = self.project_to_bev_frame(pose_world_frame=self._origin_in_world)
        origin_xy_in_world = self._origin_in_world[:2]

        # pose_in_world = [48, 1]
        pose_in_map, map_shape = self.project_to_bev_frame(pose_world_frame=pose_in_world)
        print(f"pose_in_world: {pose_in_world}")
        print(f"pose_in_map: {pose_in_map}")
        print(f"origin_in_map: {origin_in_map}")
        # time.sleep(123)

        # Flip vertically and horizontally
        origin_flipped_in_map = [origin_in_map[0], map_shape[1] - origin_in_map[1] - 1]
        # print(f"origin_xy_in_world: {origin_xy_in_world}")
        # print(f"map_pos: {origin_in_map}")
        # print(f"goal_in_world: {pose_in_world}")

        distance_on_map_world_frame = (np.asarray(pose_in_world) - np.asarray(origin_xy_in_world)) / self._resolution

        # Under BEV Frame
        if bev_frame:
            pose_in_map, map_shape = self.project_to_bev_frame(pose_world_frame=pose_in_world)
        # Human Readable Frame
        else:
            rot_mat = np.array([[1, 0],
                                [0, -1]])  # World to Human-readable Frame
            pose_in_map = origin_flipped_in_map + rot_mat @ distance_on_map_world_frame

        # pose_in_map = [pose_in_map_x, pose_in_map_y]
        # print(f"distance_in_map: {distance_in_map}")
        # print(f"origin_flipped_in_map: {origin_flipped_in_map}")
        # print(f"pose_in_map: {pose_in_map}")

        return pose_in_map

    def map_to_world_frame(self, pose_in_map, env_idx=0):
        origin_in_world, flag = self._robot.camera_sensor[env_idx].get_depth_origin_world_frame()
        if flag:
            self._origin_in_world = origin_in_world
        origin_in_map, map_shape = self.project_to_bev_frame(pose_world_frame=self._origin_in_world)

        origin_xy_in_world = self._origin_in_world[:2]

        distance_in_map = (np.asarray(pose_in_map) - np.asarray(origin_in_map)) * self._resolution

        pose_in_world = np.array([[1, 0],
                                  [0, 1]]) @ distance_in_map + origin_xy_in_world

        return pose_in_world

    # def world_to_map_frame(self, pose_in_world, env_idx=0):
    #     origin_in_world = self._robot.camera_sensor[env_idx].get_depth_origin_world_frame()
    #     origin_in_map, map_shape = self.project_to_bev_frame(pose_world_frame=origin_in_world)
    #
    #     origin_xy_in_world = origin_in_world[:2]
    #     print(f"origin_in_world_raw: {origin_in_world}")
    #     print(f"map_pos_raw: {origin_in_map}")
    #     print(f"origin_in_map: {origin_in_map}")
    #
    #     pose_in_map, _ = self.project_to_bev_frame(pose_world_frame=pose_in_world)
    #
    #     # Flip vertically
    #     # map_pos_flipped = [origin_in_map[0], map_shape[0] - origin_in_map[1] - 1]
    #     # print(f"map_pos_flipped: {map_pos_flipped}")
    #     # print(f"origin_xy_in_world: {origin_xy_in_world}")
    #     # print(f"map_pos: {origin_in_map}")
    #     # print(f"goal_in_world: {pose_in_world}")
    #     #
    #     # distance_in_map = (np.asarray(pose_in_world) - np.asarray(origin_xy_in_world)) / self._resolution
    #     # distance_in_map[1] *= -1
    #     # print(f"distance_in_map: {distance_in_map}")
    #     # pose_in_map = map_pos_flipped + distance_in_map
    #
    #     return pose_in_map

    def get_costmap(self, goal_in_map, env_idx=0, show_map=False):
        """Return the generated costmap from occupancy_map and goal.

        Parameters
        ----------
        goal_in_map : 2d array/list
                      target position (BEV frame) on the map with goal = (target_x, target_y)

        env_idx: The envs index for obtaining the robot camera (by default 0)
        show_map: whether to show the costmap

        """
        occupancy_map = self._robot.camera_sensor[env_idx].get_bev_map(as_occupancy=True, show_map=False)
        _planner = FMMPlanner(occupancy_map, resolution=self._resolution)
        costmap, costmap_for_plot = _planner.set_goal(goal_in_map)
        np.savetxt(f"costmap.txt", costmap, fmt="%.5f")

        # Save CostMap
        label_save_folder = '.'
        save_path = os.path.join(label_save_folder, f"costmap.png")
        plt.imsave(save_path, costmap_for_plot)
        save_path = os.path.join(label_save_folder, f"costmap_flip_ud.png")
        plt.imsave(save_path, np.flipud(costmap_for_plot))

        # Figure Plot
        if show_map:
            plt.subplot(1, 2, 1)
            plt.imshow(occupancy_map)
            plt.scatter(*goal_in_map[::-1], c='red', s=20, label='Goal')
            # plt.gca().invert_yaxis()  # Inverse y-axis
            # plt.gca().invert_xaxis()  # Inverse x-axis
            # plt.gca().xaxis.tick_right()  # move y-axis to the right
            plt.colorbar(label="Occupancy")
            plt.title("Occupancy Map")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.imshow(costmap_for_plot, cmap='viridis')
            # plt.imshow(costmap_for_plot, cmap='viridis')
            plt.scatter(*goal_in_map[::-1], c='red', s=20, label='Goal')
            plt.gca().invert_yaxis()  # Inverse y-axis
            plt.gca().invert_xaxis()  # Inverse x-axis
            plt.gca().yaxis.tick_right()  # move y-axis to the right
            plt.title("Cost Map")
            plt.colorbar(label="Cost", cmap='jet')
            plt.legend()
            plt.show()

        return costmap, costmap_for_plot

    def _get_bev_map(self, env_idx=0, show_map=False, reverse_xy=True, as_occupancy=False):
        bev_map = self._robot.camera_sensor[env_idx].get_bev_map(show_map=show_map, reverse_xy=reverse_xy,
                                                                 as_occupancy=as_occupancy)
        return bev_map

    def w2m(self, world_frame_pos, map_orig_pos_w):
        """translate the coordinates in world frame to the map frame"""
        return (world_frame_pos - map_orig_pos_w) / self._planner.resolution

    def generate_ref_trajectory(self, map_goal, occupancy_map=None, env_idx=0):
        if occupancy_map is None:
            occupancy_map = self._get_bev_map(as_occupancy=True, show_map=True)
        self._planner = FMMPlanner(occupancy_map, resolution=self._resolution)
        self._planner.set_goal(map_goal)

        # Position, Velocity and Yaw in World frame
        curr_pos_w = self._robot.base_position[env_idx, :2].cpu().numpy()
        yaw = self._robot.base_orientation_rpy[env_idx, 2].cpu().numpy()
        curr_vel_w = self._robot.base_velocity_world_frame[env_idx, :2].cpu().numpy()

        # Position and Velocity in Map coordinates
        curr_pos_m = self.world_to_map_frame(pose_in_world=curr_pos_w, bev_frame=True)
        curr_vel_m = curr_vel_w / self._planner.resolution
        print(f"curr_pos_w: {curr_pos_w}")
        print(f"curr_pos_m: {curr_pos_m}")
        print(f"curr_vel_w: {curr_vel_w}")
        print(f"curr_vel_m: {curr_vel_m}")
        print(f"yaw: {yaw}")

        next_pos_l, next_vel_l, stop = self._planner.get_short_term_goal(pos=curr_pos_m, yaw=yaw,
                                                                         lin_speed=np.linalg.norm(curr_vel_m))
        curr_pos_l = np.stack([curr_pos_m] * next_pos_l.shape[0])
        curr_vel_l = np.stack([curr_vel_m] * next_pos_l.shape[0])

        print(f"next_pos_l: {next_pos_l}")
        print(f"next_vel_l: {next_vel_l}")

        next_pos_trajs = self._spine.fit_eval(curr_pos_l, next_pos_l, curr_vel_l, next_vel_l)
        self.next_pos_trajs = next_pos_trajs
        self.next_pos_l = next_pos_l

        print(f"next_pos_trajs: {next_pos_trajs}")
        print(f"next_pos_trajs.shape: {next_pos_trajs.shape}")
        best_traj_idx = self._planner.find_argmin_traj(next_pos_trajs)

        next_pos, next_vel = next_pos_l[best_traj_idx], next_vel_l[best_traj_idx]
        print(f"next_pos: {next_pos}")
        print(f"next_vel: {next_vel}")
        # time.sleep(123)
        # next_pos = next_pos[::-1]
        # next_vel = next_vel[::-1]

        # curr_pos_m in world
        # curr_pos_m = np.array([[-1, 0],
        #                        [0, 1]]) @ curr_pos_m

        # curr_pos_v in world
        # curr_vel_m = np.array([[-1, 0],
        #                        [0, 1]]) @ curr_vel_m

        # next_pos in world
        # next_pos = np.array([[-1, 0],
        #                      [0, 1]]) @ next_pos

        # next_vel in world
        # next_vel = np.array([[-1, 0],
        #                      [0, 1]]) @ next_vel

        xs_raw, us_raw = self._spine.fit_reference(curr_pos_m, next_pos, curr_vel_m, next_vel)
        print(f"xs_raw: {xs_raw}")
        print(f"us_raw: {us_raw}")
        # time.sleep(5)

        return xs_raw, us_raw, stop

    def get_shortest_path(self, distance_map, start_pt, goal_pt):
        """Given the start and goal point on a distance map, find the shortest path through back-tracing

        Parameters
        ----------
        distance_map : 2d array/list
                       a gridmap with corresponding distances at each grid to the goal point. The obstacles
                       on the map should be assigned a large distance value

        start_pt: The coordinate of a start point on the map, must be integer with order in (y, x)
        goal_pt: The coordinate of a goal point on the map, must be integer with order in (y, x)

        Returns
        ----------
        shortest_path: a list of all the points from start to the goal, with each point order in (y, x)
        """
        print(f"Searching for the shortest path...")

        # Assure the point of Int type
        start_pt = (int(start_pt[0]), int(start_pt[1]))
        goal_pt = (int(goal_pt[0]), int(goal_pt[1]))

        # Gradient of the map (∇distance_map)
        gy, gx = np.gradient(distance_map)
        precise = 0.01
        gy += np.random.uniform(-precise, precise, size=gy.shape)
        gx += np.random.uniform(-precise, precise, size=gx.shape)
        np.savetxt("gx.txt", gx)
        np.savetxt("gy.txt", gy)

        shortest_path = [start_pt]
        current = start_pt
        max_cnt = 1000
        cnt = 0
        while np.linalg.norm(np.array(current) - np.array(goal_pt)) > 1:
            # Local minima
            if cnt >= max_cnt:
                break
            # print(f"current: {current}")
            y, x = current
            dy, dx = gy[y, x], gx[y, x]  # map gradient

            # Avoid local-minima
            if dy == 0 and dx == 0:
                break

            # Normalized direction
            step = np.array([-dy, -dx]) / np.linalg.norm([dy, dx]) + 1e-6
            step = [step[0].item(), step[1].item()]

            # Move to the next point and continue tracing
            next_point = [int(round(y + step[0])), int(round(x + step[1]))]

            # Exit when tracing to the start point
            if next_point == current:
                break

            # Otherwise continue tracing
            current = next_point

            # Append next point to the result
            shortest_path.append(next_point)

            cnt += 1

        return shortest_path

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, new_goal):
        print(f"Change previous goal: {self._goal} to new goal: {new_goal}")
        self._goal = new_goal
