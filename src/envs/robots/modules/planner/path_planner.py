import logging
import os
import time
from collections import deque

import numpy as np
from isaacgym import gymapi, gymutil
from numpy.core.fromnumeric import mean
from numpy.lib import utils
from matplotlib import pyplot as plt
from src.envs.robots.modules.planner.FMM import FMMPlanner
from src.envs.robots.modules.planner.spline import Spline
from src.envs.robots.modules.planner.utils import get_shortest_path, path_plot

GOALS = [(49, 1.3), (52, -0.65), (55, -1.8), (58, 0.5)]


class PathPlanner:
    def __init__(self,
                 robot,
                 sim,
                 viewer,
                 num_envs,
                 device="cuda"):
        self._robot = robot
        self._sim = sim
        self._gym = gymapi.acquire_gym()
        self._viewer = viewer
        self._num_envs = num_envs
        self._device = device

        self._goal = None
        self._resolution = 0.015
        self._planner = None
        self._spine = Spline()
        self._origin_in_world = np.array([0., 0., 0.])

        # Stored map for navigation
        self._occupancy_map = None
        self._costmap = None
        self._costmap_for_plot = None

        # Planning stuff
        self.planning_flag = True
        self.nav_stop_flag = False
        self.nav_goal_in_world = deque(GOALS)
        self._ref_trajectory = deque()

    def reset(self):
        # self._occupancy_map = None
        # self._costmap = None
        # self._costmap_for_plot = None
        self._goal = None

        # Planning stuff
        self.planning_flag = True
        self.nav_stop_flag = False
        self.nav_goal_in_world = deque(GOALS)
        self._ref_trajectory = deque()

        self.clear_goals()
        self.reset_goals()

    def init_map(self):
        map_goal = self.world_to_map_frame(pose_in_world=self._goal)
        self._occupancy_map = self._robot.camera_sensor[0].get_bev_map(as_occupancy=True,
                                                                       show_map=False,
                                                                       save_map=False)
        self._costmap, self._costmap_for_plot = self.get_costmap(goal_in_map=map_goal,
                                                                 show_map=False,
                                                                 save_map=False)

        from scipy.ndimage import gaussian_filter

        # sigma = 5
        # self._costmap = gaussian_filter(self._costmap, sigma=sigma)
        # self._costmap_for_plot = gaussian_filter(self._costmap_for_plot, sigma=sigma)

        # curr_pos_w = self._robot.base_position[0, :2]
        # start_pt = self.world_to_map_frame(pose_in_world=curr_pos_w.cpu())
        # start_pt = (int(start_pt[0]), int(start_pt[1]))
        # print(f"start_pt: {start_pt}")
        #
        # path = get_shortest_path(distance_map=self._costmap, start_pt=start_pt,
        #                          goal_pt=self._goal)
        # path_in_world = []
        # for i in range(len(path)):
        #     path_pts_in_world = self.map_to_world_frame(path[i])
        #     path_in_world.append(path_pts_in_world)
        #
        # path_plot(distance_map=self._costmap_for_plot, path=path, start=start_pt,
        #           goal=map_goal)  # Plot the shortest path

        # Plot the shortest path in the world frame
        # self._draw_path(pts=np.asarray(path_in_world))

    def reset_goals(self):
        # Draw all the goals in the world frame
        for goal in self.nav_goal_in_world:
            self._draw_goals(goal=goal, color=(1, 0, 0), size=0.17)  # Use Red to draw all goals

        if self._goal is None and len(self.nav_goal_in_world) > 0:
            self._goal = self.nav_goal_in_world.popleft()
            self._draw_goals(goal=self._goal, color=(0, 0, 1), size=0.17)  # Blue represents the current goal

    def project_to_bev_frame(self,
                             pose_world_frame,
                             grid_size=None,
                             roi_x_range=(45, 60),
                             roi_y_range=(-6, 6)):
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

        pose_in_map, map_shape = self.project_to_bev_frame(pose_world_frame=pose_in_world)

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

    def get_costmap(self, goal_in_map, occupancy_map=None, env_idx=0, show_map=False, save_map=False):
        """Return the generated costmap from occupancy_map and goal.

        Parameters
        ----------
        goal_in_map : 2d array/list
                      target position (BEV frame) on the map with goal = (target_x, target_y)
        occupancy_map: input occupancy_map or not
        env_idx: The envs index for obtaining the robot camera (by default 0)
        show_map: whether to show the cost map
        save_map: whether to save the cost map

        """
        if occupancy_map is None:
            occupancy_map = self._robot.camera_sensor[env_idx].get_bev_map(as_occupancy=True, show_map=False)
        _planner = FMMPlanner(occupancy_map, resolution=self._resolution)
        costmap, costmap_for_plot = _planner.set_goal(goal_in_map)

        # Save CostMap
        if save_map:
            label_save_folder = '.'
            save_path = os.path.join(label_save_folder, f"costmap.png")
            plt.imsave(save_path, costmap_for_plot)
            save_path = os.path.join(label_save_folder, f"costmap_flip_ud.png")
            plt.imsave(save_path, np.flipud(costmap_for_plot))
            np.savetxt(f"costmap.txt", costmap, fmt="%.5f")

        # Figure Plot
        if show_map:
            # plt.subplot(1, 2, 1)
            # plt.imshow(occupancy_map)
            # plt.scatter(*goal_in_map[::-1], c='red', s=40, label='Goal')
            # # plt.gca().invert_yaxis()  # Inverse y-axis
            # # plt.gca().invert_xaxis()  # Inverse x-axis
            # # plt.gca().xaxis.tick_right()  # move y-axis to the right
            # plt.colorbar(label="Occupancy")
            # plt.title("Occupancy Map")
            # plt.legend()

            plt.subplot(1, 2, 2)
            plt.imshow(costmap_for_plot, cmap='viridis')
            # plt.imshow(costmap_for_plot, cmap='viridis')
            plt.scatter(*goal_in_map[::-1], c='red', s=40, label='Goal')
            plt.gca().invert_yaxis()  # Inverse y-axis
            plt.gca().invert_xaxis()  # Inverse x-axis
            plt.gca().yaxis.tick_right()  # move y-axis to the right
            plt.title("Cost Map")
            plt.colorbar(label="Cost", cmap='jet')
            plt.legend(fontsize=20)
            plt.savefig("scatter_plot.png", dpi=300, bbox_inches='tight')
            plt.show()

        return costmap, costmap_for_plot

    def _get_bev_map(self, env_idx=0, show_map=False, reverse_xy=True, as_occupancy=False):
        bev_map = self._robot.camera_sensor[env_idx].get_bev_map(show_map=show_map, reverse_xy=reverse_xy,
                                                                 as_occupancy=as_occupancy)
        return bev_map

    def motion_planning(self, map_goal, occupancy_map=None, env_idx=0):
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
        # print(f"curr_pos_w: {curr_pos_w}")
        # print(f"curr_pos_m: {curr_pos_m}")
        # print(f"curr_vel_w: {curr_vel_w}")
        # print(f"curr_vel_m: {curr_vel_m}")

        next_pos_l, next_vel_l, stop = self._planner.get_short_term_goal(pos=curr_pos_m, yaw=yaw,
                                                                         lin_speed=np.linalg.norm(curr_vel_m))
        curr_pos_l = np.stack([curr_pos_m] * next_pos_l.shape[0])
        curr_vel_l = np.stack([curr_vel_m] * next_pos_l.shape[0])

        # print(f"next_pos_l: {next_pos_l}")
        # print(f"next_vel_l: {next_vel_l}")

        next_pos_trajs = self._spine.fit_eval(curr_pos_l, next_pos_l, curr_vel_l, next_vel_l)
        self.next_pos_trajs = next_pos_trajs
        self.next_pos_l = next_pos_l

        # print(f"next_pos_trajs: {next_pos_trajs}")
        # print(f"next_pos_trajs.shape: {next_pos_trajs.shape}")
        best_traj_idx = self._planner.find_argmin_traj(next_pos_trajs)

        next_pos, next_vel = next_pos_l[best_traj_idx], next_vel_l[best_traj_idx]
        # print(f"next_pos: {next_pos}")
        # print(f"next_vel: {next_vel}")

        xs_raw, us_raw = self._spine.fit_reference(curr_pos_m, next_pos, curr_vel_m, next_vel)
        # print(f"xs_raw: {xs_raw}")
        # print(f"us_raw: {us_raw}")

        return xs_raw, us_raw, stop

    def get_reference_trajectory(self):
        """Obtain generated reference motion trajectory (vx and wz)"""
        if len(self._ref_trajectory) == 0:
            # Generate trajectory (yaw_rate)

            ref_pos, ref_vel, stop_flag = self.navigate_to_goal(goal=self._goal,
                                                                occupancy_map=self._occupancy_map)
            # next_pos_trajs = []
            # for i in range(self._planner.next_pos_trajs.shape[0]):
            #     pts = self._planner.next_pos_trajs.reshape(-1, 2)[i]
            #     print(f"pts[{i}]: {pts}")
            #     pts_in_world = self._planner.map_to_world_frame(pts)
            #     next_pos_trajs.append(pts_in_world)
            #     print(f"pts_in_world: {pts_in_world}")
            # self._draw_path(pts=np.asarray(next_pos_trajs))

            # next_pos_l = []
            # print(f"self._planner.next_pos_l: {self._planner.next_pos_l}")
            # print(f"self._planner.next_pos_l: {self._planner.next_pos_l.shape}")
            # for i in range(self._planner.next_pos_l.shape[0]):
            #     pts = self._planner.next_pos_l[i]
            #     print(f"pts[{i}]: {pts}")
            #     pts_in_world = self._planner.map_to_world_frame(pts)
            #     next_pos_l.append(pts_in_world)
            #     print(f"pts_in_world: {pts_in_world}")
            # self._draw_path(pts=np.asarray(next_pos_l))

            self.nav_stop_flag = stop_flag
            self._ref_trajectory.extend(ref_vel)
        else:
            self.nav_stop_flag = self.is_curr_goal_reached()

        ret_stop_flag = self.nav_stop_flag

        # Determine to stop navigation or not
        if self.nav_stop_flag:
            current_goal = self._goal

            # Remaining navigation waypoints
            if len(self.nav_goal_in_world) > 0:
                self._goal = self.nav_goal_in_world.popleft()
                self._draw_goals(goal=self._goal, color=(0, 0, 1), size=0.17)
                map_goal = self.world_to_map_frame(pose_in_world=self._goal)
                # self._costmap, self._costmap_for_plot = self.get_costmap(goal_in_map=map_goal,
                #                                                          occupancy_map=self._occupancy_map,
                #                                                          show_map=True)
            # Has reached the final destination goal point
            else:
                self._goal = None
                self.planning_flag = False
            self._ref_trajectory.clear()
            self._ref_trajectory.append((0., 0.))
            print(f"The robot is near the goal, stop!!!")
            self._draw_goals(goal=current_goal, color=(0, 1, 0), size=0.17)

            self.nav_stop_flag = False

        ut = self._ref_trajectory.popleft()
        return ut, ret_stop_flag

    def navigate_to_goal(self, goal=None, occupancy_map=None):
        """Given the goal in the world, return reference yaw rate and stop flag"""
        if goal is None and len(self.nav_goal_in_world) > 0:
            goal = self.nav_goal_in_world[0]

        # Map the goal in world frame to map frame
        map_goal = self.world_to_map_frame(pose_in_world=goal)

        # Human readable map goal
        # x
        # ^
        # |
        # | ----> y
        logging.info(f"map_goal: {map_goal}")

        if occupancy_map is None:
            occupancy_map = self._robot.camera_sensor[0].get_bev_map(as_occupancy=True,
                                                                     show_map=False,
                                                                     save_map=False)

            from scipy.ndimage import gaussian_filter

            # sigma = 5
            # self._costmap = gaussian_filter(self._costmap, sigma=sigma)
            # self._costmap_for_plot = gaussian_filter(self._costmap_for_plot, sigma=sigma)

        ref_pos, ref_vel, stop_flag = self.motion_planning(map_goal=map_goal, occupancy_map=occupancy_map)
        return ref_pos, ref_vel, stop_flag

    def is_curr_goal_reached(self, env_idx=0):
        """Check if current goal has been reached"""
        # Position, Velocity and Yaw in World frame
        curr_pos_w = self._robot.base_position[env_idx, :2].cpu().numpy()

        # Position and Velocity in Map coordinates
        curr_pos_m = self.world_to_map_frame(pose_in_world=curr_pos_w, bev_frame=True)

        return self._planner.is_near_goal(pos_in_map=curr_pos_m)

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, new_goal):
        print(f"Change previous goal: {self._goal} to new goal: {new_goal}")
        self._goal = new_goal

    def _draw_goals(self, goal, color=(1, 0, 0), size=0.1, env_ids=0):
        # Red by default
        sphere_geom = gymutil.WireframeSphereGeometry(size, 32, 32, None, color=color)
        pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], 0), r=None)
        gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._robot.env_handles[env_ids], pose)

    def _draw_path(self, pts, color=(0, 0, 0), env_ids=0):
        # Black by default
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=color)
        for i in range(pts.shape[0]):
            pose = gymapi.Transform(gymapi.Vec3(pts[i, 0], pts[i, 1], 0), r=None)
            gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._robot.env_handles[env_ids], pose)

    def clear_goals(self):
        self._gym.clear_lines(self._viewer)
