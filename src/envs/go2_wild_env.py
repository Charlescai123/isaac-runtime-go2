"""Policy outputs desired CoM speed for Go2 to track the desired speed."""

import time
import logging
from typing import Sequence

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import to_torch
import ml_collections
import numpy as np
import torch

from src.configs.defaults import sim_config
from src.envs.go2_rewards import Go2Rewards
from src.envs.robots.modules.controller import raibert_swing_leg_controller, qp_torque_optimizer
from src.envs.robots.modules.gait_generator import phase_gait_generator
from src.envs.robots import go2_robot, go2
from src.envs.robots.modules.planner.path_planner import PathPlanner
from src.envs.robots.motors import MotorControlMode, concatenate_motor_actions
from src.envs.terrains.wild_terrain_env import WildTerrainEnv
from src.ha_teacher.ha_teacher import HATeacher
from src.coordinator.coordinator import Coordinator
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from src.utils.utils import ActionMode, RobotPusher


# @torch.jit.script
def torch_rand_float(lower, upper, shape: Sequence[int], device: str):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def gravity_frame_to_world_frame(robot_yaw, gravity_frame_vec):
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    world_frame_vec = torch.clone(gravity_frame_vec)
    world_frame_vec[:, 0] = (cos_yaw * gravity_frame_vec[:, 0] -
                             sin_yaw * gravity_frame_vec[:, 1])
    world_frame_vec[:, 1] = (sin_yaw * gravity_frame_vec[:, 0] +
                             cos_yaw * gravity_frame_vec[:, 1])
    return world_frame_vec


@torch.jit.script
def world_frame_to_gravity_frame(robot_yaw, world_frame_vec):
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    gravity_frame_vec = torch.clone(world_frame_vec)
    gravity_frame_vec[:, 0] = (cos_yaw * world_frame_vec[:, 0] +
                               sin_yaw * world_frame_vec[:, 1])
    gravity_frame_vec[:, 1] = (sin_yaw * world_frame_vec[:, 0] -
                               cos_yaw * world_frame_vec[:, 1])
    return gravity_frame_vec


def create_sim(sim_conf):
    gym = gymapi.acquire_gym()
    _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
    if sim_conf.show_gui:
        graphics_device_id = sim_device_id
    else:
        graphics_device_id = -1

    sim = gym.create_sim(sim_device_id, graphics_device_id, sim_conf.physics_engine, sim_conf.sim_params)

    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
    else:
        viewer = None

    return gym, sim, viewer


class Go2WildExploreEnv:

    def __init__(self,
                 num_envs: int,
                 config: ml_collections.ConfigDict(),
                 device: str = "cuda",
                 show_gui: bool = False,
                 use_real_robot: bool = False):

        self._num_envs = num_envs
        self._device = device
        self._show_gui = show_gui
        self._config = config
        self._gamma = config.gamma
        self._terrain_region = to_torch(config.terrain_region, device=device)
        self._init_map_flag = True

        self._use_real_robot = use_real_robot

        with self._config.unlocked():
            if self._config.get('observation_noise', None) is not None:
                self._config.observation_noise = to_torch(
                    self._config.observation_noise, device=self._device)

        # Simulation config
        self._sim_conf = sim_config.get_config(
            show_gui=show_gui,
            use_gpu=("cuda" in device),
            use_penetrating_contact=self._config.get('use_penetrating_contact', False)
        )

        self._gym, self._sim, self._viewer = create_sim(self._sim_conf)
        self._create_terrain()

        self._init_positions = self._compute_init_positions()
        if self._use_real_robot:
            robot_class = go2_robot.Go2Robot
        else:
            robot_class = go2.Go2

        # The Robot Env
        self._robot = robot_class(
            num_envs=self._num_envs,
            init_positions=self._init_positions,
            sim=self._sim,
            viewer=self._viewer,
            world_env=WildTerrainEnv,
            sim_config=self._sim_conf,
            motor_control_mode=MotorControlMode.HYBRID,
            motor_torque_delay_steps=self._config.get('motor_torque_delay_steps', 0)
        )

        strength_ratios = self._config.get('motor_strength_ratios', 0.7)
        if isinstance(strength_ratios, Sequence) and len(strength_ratios) == 2:
            ratios = torch_rand_float(lower=to_torch([strength_ratios[0]], device=self._device),
                                      upper=to_torch([strength_ratios[1]], device=self._device),
                                      shape=(self._num_envs, 3),
                                      device=self._device)
            # Use the same ratio for all ab/ad motors, all hip motors, all knee motors
            ratios = torch.concatenate((ratios, ratios, ratios, ratios), dim=1)
            self._robot.motor_group.strength_ratios = ratios
        else:
            self._robot.motor_group.strength_ratios = strength_ratios

        # Need to set frictions twice to make it work on GPU... 😂
        self._robot.set_foot_frictions(0.01)
        self._robot.set_foot_frictions(self._config.get('foot_friction', 1.))

        # Coordinator
        self.coordinator = Coordinator(num_envs=self._num_envs, device=self._device)

        # HA-Teacher
        self.ha_teacher = HATeacher(num_envs=self._num_envs, teacher_cfg=config.ha_teacher, device=self._device)

        # Gait scheduler
        self._gait_generator = phase_gait_generator.PhaseGaitGenerator(self._robot, self._config.gait)

        # Reference state
        self.desired_vx = config.desired_vx
        self.desired_com_height = config.desired_pz
        self.desired_wz = config.desired_wz
        self.clip_wz = config.clip_wz

        # Swing leg controller
        self._swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self._robot,
            self._gait_generator,
            desired_base_height=self.desired_com_height,
            foot_height=self._config.get('swing_foot_height', 0.12),
            foot_landing_clearance=self._config.get('swing_foot_landing_clearance', 0.02)
        )

        # Stance controller
        self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            self._robot,
            base_position_kp=self._config.get('base_position_kp', np.array([0., 0., 50.])),
            base_position_kd=self._config.get('base_position_kd', np.array([10., 10., 10.])),
            base_orientation_kp=self._config.get('base_orientation_kp', np.array([50., 50., 0.])),
            base_orientation_kd=self._config.get('base_orientation_kd', np.array([10., 10., 10.])),
            acc_lb=self._config.get('action_lb', np.array([-10, -10, -10, -20, -20, -20])),
            acc_ub=self._config.get('action_ub', np.array([10, 10, 10, 20, 20, 20])),
            weight_ddq=self._config.get('qp_weight_ddq', np.diag([20.0, 20.0, 5.0, 1.0, 1.0, .2])),
            foot_friction_coef=self._config.get('qp_foot_friction_coef', 0.7),
            clip_grf=self._config.get('clip_grf_in_sim') or self._use_real_robot,
            # body_inertia=self._config.get('qp_body_inertia', np.diag([0.14, 0.35, 0.35]) * 0.5),
            use_full_qp=self._config.get('use_full_qp', False)
        )

        # Quadruped Robot Pusher
        self._pusher = RobotPusher(
            robot=self._robot,
            sim=self._sim,
            viewer=self._viewer,
            num_envs=self._num_envs,
            device=self._device
        )
        self._push_flag = False

        # Set reference trajectory
        self._torque_optimizer.set_controller_reference(desired_height=self.desired_com_height,
                                                        desired_lin_vel=[self.desired_vx, 0, 0],
                                                        desired_rpy=[0., 0., 0.],
                                                        desired_ang_vel=[0., 0., self.desired_wz])
        # Path Planner
        self._planner = PathPlanner(robot=self._robot,
                                    sim=self._sim,
                                    viewer=self._viewer,
                                    num_envs=self._num_envs,
                                    device=self._device)
        self.sub_goal_reach_flag = False
        self.sub_goal_reach_time = []

        # Episodic statistics
        self._go2_reward = Go2Rewards(self, reward_cfg=config.reward)
        self._step_count = torch.zeros(self._num_envs, device=self._device)
        self._episode_length = self._config.episode_length_s / self._config.env_dt
        self._construct_observation_and_action_space()

        self._privileged_obs_buf = None
        self._obs_buf = torch.zeros((self._num_envs, config.obs_dim), device=self._device)
        self._last_obs_buf = torch.zeros((self._num_envs, config.obs_dim), device=self._device)
        self._last_action = torch.zeros((self._num_envs, config.act_dim), device=self._device)
        self._last_terminal_action = torch.zeros((self._num_envs, config.act_dim), device=self._device)
        self._goal_position = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._cycle_count = torch.zeros(self._num_envs, device=self._device)
        self._teacher_activation_times = torch.zeros(self._num_envs, device=self._device)
        self._student_activation_times = torch.zeros(self._num_envs, device=self._device)
        self.last_goal_distance = self.goal_distance
        # self._motor_energy_consumption = []

        self._extras = dict()
        self._extras["fails"] = 0

        # Running a few steps with dummy commands to ensure JIT compilation
        if self._num_envs == 1 and self._use_real_robot:
            for state in range(16):
                desired_contact_state = torch.tensor(
                    [[(state & (1 << i)) != 0 for i in range(4)]], dtype=torch.bool, device=self._device)
                for _ in range(3):
                    self._gait_generator.update()
                    self._swing_leg_controller.update()
                    desired_foot_positions = self._swing_leg_controller.desired_foot_positions
                    self._torque_optimizer.get_action(
                        desired_contact_state, swing_foot_position=desired_foot_positions)

    def _create_terrain(self):
        """Creates terrains.

        Note that we set the friction coefficient to all 0 here. This is because
        Isaac seems to pick the larger friction out of a contact pair as the
        actual friction coefficient. We will set the corresponding friction coefficient
        in robot friction.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = .2
        plane_params.dynamic_friction = .2
        plane_params.restitution = 0.
        self._gym.add_ground(self._sim, plane_params)
        self._terrain = None

    def _compute_init_positions(self):
        init_positions = torch.zeros((self._num_envs, 3), device=self._device)
        num_cols = int(np.sqrt(self._num_envs))
        distance = 1.
        for idx in range(self._num_envs):
            init_positions[idx, 0] = idx // num_cols * distance
            init_positions[idx, 1] = idx % num_cols * distance
            # init_positions[idx, 2] = 0.268
            init_positions[idx, 2] = 0.3
        return to_torch(init_positions, device=self._device)

    def _construct_observation_and_action_space(self):
        self._observation_lb = to_torch(self._config.observation_lb, device=self._device)
        self._observation_ub = to_torch(self._config.observation_ub, device=self._device)
        self._action_lb = to_torch(self._config.action_lb, device=self._device)
        self._action_ub = to_torch(self._config.action_ub, device=self._device)

    def reset(self) -> torch.Tensor:
        return self.reset_idx(torch.arange(self._num_envs, device=self._device))

    def reset_idx(self, env_ids) -> torch.Tensor:
        # Aggregate rewards
        self._extras["time_outs"] = self._episode_terminated()
        if env_ids.shape[0] > 0:
            self._extras["episode"] = {}
            # self._extras["episode"]["cycle_count"] = torch.mean(
            #     self._gait_generator.true_phase[env_ids]) / (2 * torch.pi)

            # Reset all rewards statistics
            for reward_name in self._go2_reward.episode_sums.keys():
                self._go2_reward.episode_sums[reward_name][env_ids] = 0

            self._step_count[env_ids] = 0
            self._cycle_count[env_ids] = 0

            self._robot.reset_idx(env_ids)
            self._swing_leg_controller.reset_idx(env_ids)
            self._gait_generator.reset_idx(env_ids)
            self._planner.reset()
            self._goal_position[env_ids, :2] = to_torch(self._planner.goal, dtype=torch.float32, device=self._device)

            self._obs_buf = self._get_observations()
            self._privileged_obs_buf = self._get_privileged_observations()
            self._last_obs_buf = torch.zeros((self._num_envs, self._config.obs_dim), device=self._device)
            self._last_action = torch.zeros((self._num_envs, self._config.act_dim), device=self._device)
            self._last_terminal_action = torch.zeros((self._num_envs, self._config.act_dim), device=self._device)

        return self._obs_buf, self._privileged_obs_buf

    def step(self, drl_action: torch.Tensor):

        drl_action = torch.clip(drl_action, self._action_lb, self._action_ub)
        nominal_actions = torch.zeros(self._num_envs, 6, device=self._device)
        sum_reward = torch.zeros(self._num_envs, device=self._device)
        dones = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

        logs = []
        start = time.time()

        for step in range(max(int(self._config.env_dt / self._robot.control_timestep), 1)):
            logging.info(f"config.env_dt: {self._config.env_dt}")
            logging.info(f"self._robot.control_timestep: {self._robot.control_timestep}")
            self._gait_generator.update()
            self._swing_leg_controller.update()

            # self._robot.state_estimator.update_ground_normal_vec()
            # self._robot.state_estimator.update_foot_contact(self._gait_generator.desired_contact_state)

            # Let the planner makes planning
            if self._planner.planning_flag:

                # Do map initialize once
                if self._init_map_flag:
                    self._planner.init_map()
                    self._init_map_flag = False

                # Take the trajectory generated from the planner
                ut, self.sub_goal_reach_flag = self._planner.get_reference_trajectory()
                if self.sub_goal_reach_flag:
                    self.sub_goal_reach_time.append(self._step_count)
                vel_x = ut[0] * 1
                ref_wz = ut[1] * 1

                # Set desired command for vx and wz
                # self.desired_vx = vel_x
                self.desired_wz = np.clip(ref_wz, a_min=self.clip_wz[0], a_max=self.clip_wz[1])

                # Update goal position to calc distance
                if len(self._planner.nav_goal_in_world) > 0:
                    self._goal_position[0, :2] = to_torch(self._planner.goal, dtype=torch.float32, device=self._device)

            else:
                time.sleep(0.02)
                self.desired_vx = 0.
                self.desired_wz = 0.

            # T = 0.02
            # yaw = self._robot.base_orientation_rpy[0, 2]
            # self.desired_yaw = yaw + T * self.desired_wz
            # Setup controller reference
            self._torque_optimizer.set_controller_reference(
                desired_height=self.desired_com_height,
                desired_lin_vel=[self.desired_vx, 0, 0],
                desired_rpy=[0, 0, 0],
                desired_ang_vel=[0, 0, self.desired_wz]
            )

            if self._use_real_robot:
                self._robot.state_estimator.update_foot_contact(
                    self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error
                self._robot.update_desired_foot_contact(
                    self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error

            # Compute model-based acceleration
            self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = self._torque_optimizer.compute_model_acc(
                foot_contact_state=self._gait_generator.desired_contact_state,
                desired_foot_position=self._swing_leg_controller.desired_foot_positions
            )

            # HP-Student action (residual form)
            raw_weights = to_torch([1.5, 0.5, 0.5, 0.1, 0.1, 0.5], device=self._device)
            weights = self._gamma
            drl_action_res = drl_action * weights
            phy_action_res = self._solved_acc * (1 - weights)
            hp_action = drl_action_res + phy_action_res
            # print(f"drl_action_res: {drl_action_res}")
            # print(f"phy_action_res: {phy_action_res}")
            # print(f"hp_action: {hp_action}")
            # print(f"drl action: {drl_action}")
            # import pdb
            # pdb.set_trace()

            # HA-Teacher update
            self._robot.energy_2d = self.ha_teacher.update(self._torque_optimizer.tracking_error)

            # HA-Teacher action
            ha_action, dwell_flag = self.ha_teacher.get_action()

            self._terminal_stance_ddq, action_mode = self.coordinator.get_terminal_action(
                hp_action=hp_action,
                ha_action=ha_action,
                plant_state=self._torque_optimizer.tracking_error,
                safety_subset=self.ha_teacher.safety_subset,
                dwell_flag=dwell_flag
            )
            logging.info(f"hp_action: {hp_action}")
            logging.info(f"ha_action: {ha_action}")
            logging.info(f"self._torque_optimizer.tracking_error: {self._torque_optimizer.tracking_error}")
            logging.info(f"terminal_stance_ddq: {self._terminal_stance_ddq}")
            logging.info(f"action_mode: {action_mode}")

            # Action mode indices
            hp_indices = torch.argwhere(action_mode == ActionMode.STUDENT.value).squeeze(-1)
            ha_indices = torch.argwhere(action_mode == ActionMode.TEACHER.value).squeeze(-1)
            hp_motor_action = None
            ha_motor_action = None

            logging.info(f"hp_indices: {hp_indices}")
            logging.info(f"ha_indices: {ha_indices}")

            # HP-Student in Control
            if len(hp_indices) > 0:
                hp_motor_action, self._desired_acc[hp_indices], self._solved_acc[hp_indices], \
                    self._qp_cost[hp_indices], self._num_clips[hp_indices] = self._torque_optimizer.get_action(
                    foot_contact_state=self._gait_generator.desired_contact_state[hp_indices],
                    swing_foot_position=self._swing_leg_controller.desired_foot_positions[hp_indices],
                    generated_acc=self._terminal_stance_ddq[hp_indices]
                )
                self._student_activation_times[hp_indices] += 1
                self.robot.set_robot_base_color(color=(0, 0, 1.), env_ids=hp_indices)  # Display Blue
                nominal_actions[hp_indices] = drl_action  # Nominal actions for HP-student

            # HA-Teacher in Control
            if len(ha_indices) > 0:
                ha_motor_action, self._desired_acc[ha_indices], self._solved_acc[ha_indices], \
                    self._qp_cost[ha_indices], self._num_clips[ha_indices] = self._torque_optimizer.get_action(
                    foot_contact_state=self._gait_generator.desired_contact_state[ha_indices],
                    swing_foot_position=self._swing_leg_controller.desired_foot_positions[ha_indices],
                    generated_acc=self._terminal_stance_ddq[ha_indices],
                    gravity_frame=True
                )
                self._teacher_activation_times[ha_indices] += 1
                self.robot.set_robot_base_color(color=(1., 0, 0), env_ids=ha_indices)  # Display Red
                # Nominal actions for HP-student
                nominal_actions[ha_indices] = (ha_action[ha_indices] - phy_action_res[ha_indices]) / weights

            # Unknown Action Mode
            if len(hp_indices) == 0 and len(ha_indices) == 0:
                raise RuntimeError(f"Unrecognized Action Mode: {action_mode}")

            # Add both HP-Student and HA-Teacher Motor Action
            motor_action = concatenate_motor_actions(command1=hp_motor_action, indices1=hp_indices,
                                                     command2=ha_motor_action, indices2=ha_indices)
            # Logging
            logs.append(
                dict(timestamp=self._robot.time_since_reset,
                     base_position=torch.clone(self._robot.base_position),
                     base_orientation_rpy=torch.clone(
                         self._robot.base_orientation_rpy),
                     base_velocity=torch.clone(self._robot.base_velocity_body_frame),
                     base_angular_velocity=torch.clone(
                         self._robot.base_angular_velocity_body_frame),
                     motor_positions=torch.clone(self._robot.motor_positions),
                     motor_velocities=torch.clone(self._robot.motor_velocities),
                     motor_action=motor_action,
                     motor_torques=self._robot.motor_torques,
                     motor_power=self._robot.motor_power,
                     num_clips=self._num_clips,
                     foot_contact_state=self._gait_generator.desired_contact_state,
                     foot_contact_force=self._robot.foot_contact_forces,
                     desired_swing_foot_position=self._swing_leg_controller.desired_foot_positions,
                     desired_acc_body_frame=self._desired_acc,
                     desired_vx=self.desired_vx,
                     desired_wz=self.desired_wz,
                     desired_com_height=self.desired_com_height,
                     ha_action=ha_action,
                     hp_action=hp_action,
                     action_mode=action_mode,
                     acc_min=self._action_lb,
                     acc_max=self._action_ub,
                     lyapunov_energy=to_torch(self._robot.energy_2d, device=self._device),
                     solved_acc_body_frame=self._solved_acc,
                     foot_positions_in_base_frame=self._robot.foot_positions_in_base_frame,
                     env_action=drl_action,
                     env_obs=torch.clone(self._obs_buf)
                     )
            )

            if self._use_real_robot:
                logs[-1]["base_acc"] = np.array(
                    self._robot.raw_state.imu.accelerometer)  # pytype: disable=attribute-error

            # Error in last step
            err_prev = self._torque_optimizer.tracking_error

            #################################   Step The Motor Action   #################################
            self._robot.step(motor_action)
            #############################################################################################

            # Motor energy consumption
            # self._motor_energy_consumption.append(self.get_motor_energy_consumption())

            self._obs_buf = self._get_observations()
            self._privileged_obs_buf = self.get_privileged_observations()

            # Delta action for smoothness
            delta_action = torch.sqrt(torch.sum(
                torch.square(self._last_terminal_action - self._terminal_stance_ddq), dim=1))

            # Error in next step
            err_next = self._torque_optimizer.tracking_error

            # Dones or not
            is_done, is_fail = self._is_done()
            dones = torch.logical_or(dones, is_done)

            # Get Total reward
            rewards = self._go2_reward.compute_reward(err_prev, err_next, is_fail, delta_action)

            # Episodic statistics
            self.update_episodic_statistics(env_ids=torch.arange(self._num_envs, device=self._device))

            # Sum reward
            sum_reward += rewards * torch.logical_not(dones)

            # Push indicator
            if self._pusher.indicator_flag:
                if self._pusher.indicator_cnt < self._pusher.indicator_max:
                    self._pusher.indicator_cnt += 1
                else:
                    self._pusher.clear_indicator()

            # Monitor the pusher
            self._push_flag = self._pusher.monitor_push(step_cnt=self._step_count,
                                                        env_ids=torch.arange(self._num_envs, device=self._device))

            self._extras["logs"] = logs

            # Resample commands
            # new_cycle_count = (self._gait_generator.true_phase / (2 * torch.pi)).long()
            # finished_cycle = new_cycle_count > self._cycle_count
            # self._cycle_count = new_cycle_count

            # is_terminal = torch.logical_or(finished_cycle, dones)
            # if is_terminal.any():
            #     print(f"terminal_reward is: {self.get_terminal_reward(is_terminal, dones)}")

            if not self._use_real_robot:
                self.reset_idx(dones.nonzero(as_tuple=False).flatten())
                pass

            if self._show_gui:
                self._robot.render()

        self._step_count += 1
        self._last_obs_buf = torch.clone(self._obs_buf)
        self._last_action = torch.clone(drl_action)
        self._last_terminal_action = torch.clone(self._terminal_stance_ddq)
        self.last_goal_distance = self.goal_distance

        end = time.time()
        logging.info(f"***************** step duration: {end - start} *****************")
        return self._obs_buf, self._privileged_obs_buf, nominal_actions, sum_reward, dones, self._extras

    def _get_observations(self):
        distance_obs = self.goal_distance.unsqueeze(dim=-1)
        yaw_diff_obs = self.goal_yaw_difference.unsqueeze(dim=-1)
        robot_obs = self._torque_optimizer.tracking_error[:, 2:]
        # desired_wz_obs = to_torch([self.desired_wz], device=self._device).unsqueeze(dim=-1)

        obs = torch.concatenate((distance_obs, yaw_diff_obs, robot_obs), dim=1)
        # if self._config.get("observation_noise", None) is not None and (not self._use_real_robot):
        #     obs += torch.randn_like(obs) * self._config.observation_noise
        return obs

    def get_observations(self):
        return self._get_observations()

    def _get_privileged_observations(self):
        return None

    def get_privileged_observations(self):
        return self._privileged_obs_buf

    def _episode_terminated(self):
        timeout = (self._step_count >= self._episode_length)
        return timeout

    def update_episodic_statistics(self, env_ids):
        # Episode count
        self._extras["episode"]["ep_step"] = self._step_count

        for reward_name in self._go2_reward.episode_sums.keys():
            # Stepwise reward
            reward_val = torch.mean(self._go2_reward.episode_sums[reward_name][env_ids]
                                    / self._step_count[env_ids])
            self._extras["episode"]["reward_{}".format(reward_name)] = torch.nan_to_num(reward_val, nan=0.0)

    def _is_terminate(self):
        """Check the robot termination"""
        is_terminate = torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device)

        # Planning termination
        if self._config.get('terminate_on_destination_reach', False):
            is_terminate = torch.logical_or(is_terminate, to_torch(not self._planner.planning_flag, dtype=torch.bool,
                                                                   device=self._device))
        # Body contact termination
        if self._config.get('terminate_on_body_contact', False):
            is_terminate = torch.logical_or(is_terminate, self._robot.has_body_contact)

        if self._config.get('terminate_on_dense_body_contact', False):
            is_terminate = torch.logical_or(is_terminate, self._robot.has_dense_body_contact)

        if self._config.get('terminate_on_limb_contact', False):
            limb_contact = torch.logical_or(self._robot.calf_contacts, self._robot.thigh_contacts)
            limb_contact = torch.sum(limb_contact, dim=1)
            is_terminate = torch.logical_or(is_terminate, limb_contact > 0)

        # Yaw deviation termination
        if self._config.get('terminate_on_yaw_deviation', False):
            yaw_diff = torch.abs(self.goal_yaw_difference)
            terminate_by_yaw = yaw_diff > self._config.yaw_deviation_threshold
            is_terminate = torch.logical_or(is_terminate, terminate_by_yaw)

        # Maximum steps
        if self._config.get('terminate_on_timeout', False):
            timeout = self._episode_terminated()
            is_terminate = torch.logical_or(is_terminate, timeout)

        # Out of terrain region
        if self._config.get('terminate_on_out_of_terrain', False):
            x_range, y_range = self._terrain_region
            curr_pos = self._robot.base_position_world[:, :2]
            in_x_range = (curr_pos[:, 0] >= x_range[0]) & (curr_pos[:, 0] <= x_range[1])
            in_y_range = (curr_pos[:, 1] >= y_range[0]) & (curr_pos[:, 1] <= y_range[1])
            in_range = in_x_range & in_y_range
            is_terminate = torch.logical_or(is_terminate, ~in_range)  # terminate when out of region

        return is_terminate

    def _is_fail(self):
        """Check the robot failure according to defined safety condition (hard constraint)"""
        gravity_threshold = 0.1
        is_fail = torch.logical_or(
            # self._robot.projected_gravity[:, 2] < gravity_threshold,
            torch.full((self._num_envs,), False, dtype=torch.bool, device=self._device),
            self._robot.base_position[:, 2] < self._config.get('terminate_on_height', 0.12))
        return is_fail

    def _is_done(self):
        """Check done or not"""
        # Check the system's failure
        is_fail = self._is_fail()
        if torch.any(is_fail):
            self._extras["fails"] += is_fail.sum()

        # Check the termination
        is_terminated = self._is_terminate()
        # if torch.any(is_terminated):
        #     import pdb
        #     pdb.set_trace()

        return torch.logical_or(is_terminated, is_fail), is_fail

    @property
    def device(self):
        return self._device

    @property
    def robot(self):
        return self._robot

    @property
    def gait_generator(self):
        return self._gait_generator

    @property
    def torque_optimizer(self):
        return self._torque_optimizer

    @property
    def goal_position(self):
        return self._goal_position

    @property
    def goal_distance(self):
        base_position = self._robot.base_position_world
        distance = torch.sqrt(
            torch.sum(torch.square(base_position[:, :2] - self._goal_position[:, :2]), dim=1))
        return distance

    @property
    def goal_yaw_difference(self):
        base_yaw_world = self._robot.base_orientation_rpy[:, 2]
        base_position_xy = self._robot.base_position_world[:, :2]
        goal_position_xy = self._goal_position[:, :2]

        delta_position_xy = goal_position_xy - base_position_xy
        goal_yaw_world = torch.atan2(delta_position_xy[:, 1], delta_position_xy[:, 0])

        # yaw_diff = goal_yaw_world - base_yaw_world

        # Wrap to [-pi, pi]
        # yaw_diff = (yaw_diff + torch.pi) % (2 * torch.pi) - torch.pi

        # Wrap to [0, pi]
        yaw_diff = torch.abs(goal_yaw_world - base_yaw_world)

        return to_torch(yaw_diff, device=self._device)

    @property
    def planner(self):
        return self._planner

    @property
    def action_space(self):
        return self._action_lb, self._action_ub

    @property
    def observation_space(self):
        return self._observation_lb, self._observation_ub

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def num_obs(self):
        return self._observation_lb.shape[0]

    @property
    def num_privileged_obs(self):
        return None

    @property
    def num_actions(self):
        return self._action_lb.shape[0]

    @property
    def max_episode_length(self):
        return self._episode_length

    @property
    def episode_length_buf(self):
        return self._step_count

    @episode_length_buf.setter
    def episode_length_buf(self, new_length: torch.Tensor):
        self._step_count = to_torch(new_length, device=self._device)
