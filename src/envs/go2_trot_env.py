"""Policy outputs desired CoM speed for Go2 to track the desired speed."""

import itertools
import time
from typing import Sequence

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import to_torch
import ml_collections
import numpy as np
import torch

from src.configs.defaults import sim_config
from src.envs.robots.controller import raibert_swing_leg_controller, phase_gait_generator
from src.envs.robots.controller import qp_torque_optimizer
from src.envs import go1_rewards
from src.envs.robots import go2_robot, go2
from src.envs.robots.motors import MotorControlMode
from src.envs.terrains.wild_env import WildTerrainEnv


def generate_seed_sequence(seed, num_seeds):
    np.random.seed(seed)
    return np.random.randint(0, 100, size=num_seeds)


random_push_sequence = generate_seed_sequence(seed=1, num_seeds=100)
dual_push_sequence = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
push_magnitude_list = [-1.0, 1.2, -1.3, 1.4, -1.5]

# For backwards
# push_delta_vel_list_x = [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]
# push_delta_vel_list_y = [-0.62, 1.25, -0.7, 0.6, -0.55, 0.6, -0.6]
# push_delta_vel_list_z = [-0.72, -0.72, -0.72, -0.72, -0.72, -0.72, -0.72]
# push_interval = np.array([300, 450, 620, 750, 820, 950, 1050, 1200]) - 1

# For forward
push_delta_vel_list_x = [0.25, 0.25, 0.25, 0.25, 0.3, 0.25, 0.25]
push_delta_vel_list_y = [-0.7, 0.75, -1.2, 0.6, -0.7, 0.7, -0.6]
push_delta_vel_list_z = [-0.72, -0.7, -0.72, -0.72, -0.72, -0.72, -0.72]
push_interval = np.array([300, 450, 620, 750, 850, 1000, 1050, 1200]) - 1


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

    print(f"self.sim_device_id: {sim_device_id}")
    print(f"self.graphics_device_id: {graphics_device_id}")
    print(f"self.physics_engine: {sim_conf.physics_engine}")
    print(f"self.sim_params: {sim_conf.sim_params}")
    sim = gym.create_sim(sim_device_id, graphics_device_id, sim_conf.physics_engine, sim_conf.sim_params)
    print(f"sim: {sim}")

    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        print(f"viewer: {viewer}")
        # time.sleep(123)
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V,
                                            "toggle_viewer_sync")
    else:
        viewer = None

    return gym, sim, viewer


class Go2TrotEnv:

    def __init__(self,
                 num_envs: int,
                 config: ml_collections.ConfigDict(),
                 device: str = "cuda",
                 show_gui: bool = False,
                 use_real_robot: bool = False):
        self._cnt = 0
        self._push_cnt = 0
        self._push_magnitude = 1

        self._env_handles = []
        self._num_envs = num_envs
        self._device = device
        self._show_gui = show_gui
        self._config = config
        # print(f"self._config: {self._config}")
        # time.sleep(123)
        self._use_real_robot = use_real_robot
        self._jumping_distance_schedule = config.get('jumping_distance_schedule', None)
        if self._jumping_distance_schedule is not None:
            self._jumping_distance_schedule = itertools.cycle(self._jumping_distance_schedule)
        with self._config.unlocked():
            self._config.goal_lb = to_torch(self._config.goal_lb, device=self._device)
            self._config.goal_ub = to_torch(self._config.goal_ub, device=self._device)
            if self._config.get('observation_noise', None) is not None:
                self._config.observation_noise = to_torch(
                    self._config.observation_noise, device=self._device)

        from src.ha_teacher.ha_teacher import HATeacher
        from src.coordinator.coordinator import Coordinator
        from omegaconf import DictConfig

        teacher_config = DictConfig(
            {"chi": 0.15, "tau": 100, "teacher_enable": True, "teacher_learn": True, "epsilon": 1,
             "cvxpy_solver": "solver"}
        )
        # coordinator_config = DictConfig({"teacher_learn": True, "max_dwell_steps": 100})
        self.ha_teacher = HATeacher(num_envs=self._num_envs, teacher_cfg=teacher_config)
        self.coordinator = Coordinator(num_envs=self._num_envs)

        # Set up robot and controller
        use_gpu = ("cuda" in device)
        self._sim_conf = sim_config.get_config(
            use_gpu=use_gpu,
            show_gui=show_gui,
            use_penetrating_contact=self._config.get('use_penetrating_contact', False)
        )

        self.desired_vx = 0.7
        self.desired_com_height = 0.3

        self._gym, self._sim, self._viewer = create_sim(self._sim_conf)
        self._create_terrain()

        # Initialize Wild Terrain Env (Must after the robot)
        # self._wild_terrain_env = WildTerrainEnv(
        #     sim=self._sim,
        #     gym=self._gym,
        #     viewer=self._viewer,
        #     num_envs=self._num_envs,
        #     env_handle=self._env_handles,
        #     sim_config=self._sim_conf
        # )

        # add_ground(self._gym, self._sim)
        # add_terrain(self._gym, self._sim, "stair")
        # add_terrain(self._gym, self._sim, "slope")
        # add_terrain(self._gym, self._sim, "stair", 3.95, True)
        # add_terrain(self._gym, self._sim, "stair", 0., True)
        print(f"self._env_handles: {self._env_handles}")

        self._indicator_flag = False
        self._indicator_cnt = 0

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

        # self._init_buffer()

        strength_ratios = self._config.get('motor_strength_ratios', 0.7)
        if isinstance(strength_ratios, Sequence) and len(strength_ratios) == 2:
            ratios = torch_rand_float(lower=to_torch([strength_ratios[0]],
                                                     device=self._device),
                                      upper=to_torch([strength_ratios[1]],
                                                     device=self._device),
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

        # 获取所有 actor 的名称
        actor_count = self._gym.get_actor_count(self._robot._envs[0])
        actor_names = [self._gym.get_actor_name(self._robot._envs[0], i) for i in range(actor_count)]

        # 打印所有 actor 的名称
        for name in actor_names:
            print(name)
        # 获取 root state tensor
        root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)

        # root_state = np.array(root_state_tensor).reshape(-1, 13)  # 13维状态向量
        # from isaacgym import gymtorch
        # print(f"root_state_tensor: {gymtorch.wrap_tensor(root_state_tensor)}")
        # for actor_index in range(self._gym.get_actor_count(self._robot._envs[0])):
        #     if self._gym.get_actor_name(self._robot._envs[0], actor_index) == "robot":
        #         print(f"Found actor in env {self._robot._envs[0]}, index {actor_index}")
        #         # 获取全局索引
        #         global_index = env_index * actors_per_env + actor_index
        #         actor_position = root_state[global_index, :3]
        #         actor_rotation = root_state[global_index, 3:7]
        #         print(f"Actor Position: {actor_position}")
        #         print(f"Actor Rotation (Quaternion): {actor_rotation}")
        # time.sleep(123)
        def get_gait_config():
            config = ml_collections.ConfigDict()
            config.stepping_frequency = 2  # 1
            config.initial_offset = np.array([0., 0.5, 0.5, 0.],
                                             dtype=np.float32) * (2 * np.pi)
            config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            # config.initial_offset = np.array([0.15, 0.15, -0.35, -0.35]) * (2 * np.pi)
            # config.swing_ratio = np.array([0.6, 0.6, 0.6, 0.6])
            return config

        self._config.gait = get_gait_config()
        # time.sleep(123)
        self._gait_generator = phase_gait_generator.PhaseGaitGenerator(
            self._robot, self._config.gait)
        self._swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self._robot,
            self._gait_generator,
            foot_height=self._config.get('swing_foot_height', 0.1),
            foot_landing_clearance=self._config.get('swing_foot_landing_clearance', 0.))
        self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            self._robot,
            base_position_kp=self._config.get('base_position_kp', np.array([0., 0., 50.])),
            base_position_kd=self._config.get('base_position_kd', np.array([10., 10., 10.])),
            base_orientation_kp=self._config.get('base_orientation_kp', np.array([50., 50., 0.])),
            base_orientation_kd=self._config.get('base_orientation_kd', np.array([10., 10., 10.])),
            weight_ddq=self._config.get('qp_weight_ddq', np.diag([20.0, 20.0, 5.0, 1.0, 1.0, .2])),
            foot_friction_coef=self._config.get('qp_foot_friction_coef', 0.7),
            clip_grf=self._config.get('clip_grf_in_sim') or self._use_real_robot,
            # body_inertia=self._config.get('qp_body_inertia', np.diag([0.14, 0.35, 0.35]) * 0.5),
            use_full_qp=self._config.get('use_full_qp', False)
        )

        self._steps_count = torch.zeros(self._num_envs, device=self._device)
        self._init_yaw = torch.zeros(self._num_envs, device=self._device)
        self._episode_length = self._config.episode_length_s / self._config.env_dt
        self._construct_observation_and_action_space()
        if self._config.get("observe_heights", False):
            self._height_points, self._num_height_points = self._compute_height_points(
            )
        self._obs_buf = torch.zeros((self._num_envs, 12), device=self._device)
        self._privileged_obs_buf = None
        self._desired_landing_position = torch.zeros((self._num_envs, 3),
                                                     device=self._device,
                                                     dtype=torch.float)
        self._cycle_count = torch.zeros(self._num_envs, device=self._device)

        self._rewards = go1_rewards.Go1Rewards(self)
        self._prepare_rewards()
        self._extras = dict()

        # self.draw_lane()

        # Running a few steps with dummy commands to ensure JIT compilation
        if self._num_envs == 1 and self._use_real_robot:
            for state in range(16):
                desired_contact_state = torch.tensor(
                    [[(state & (1 << i)) != 0 for i in range(4)]],
                    dtype=torch.bool,
                    device=self._device)
                for _ in range(3):
                    self._gait_generator.update()
                    self._swing_leg_controller.update()
                    desired_foot_positions = self._swing_leg_controller.desired_foot_positions
                    self._torque_optimizer.get_action(
                        desired_contact_state, swing_foot_position=desired_foot_positions)

    def _init_buffer(self):
        self._robot._init_buffers()
        self._robot._post_physics_step()

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

        # from ippc.quad_gym.env_builder import build_a1_ground_env
        # build_a1_ground_env()

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
        # robot_lb = to_torch(
        #     [0., -3.14, -3.14, -4., -4., -10., -3.14, -3.14, -3.14] +
        #     [-0.5, -0.5, -0.4] * 4,
        #     device=self._device)
        # robot_ub = to_torch([0.6, 3.14, 3.14, 4., 4., 10., 3.14, 3.14, 3.14] +
        #                     [0.5, 0.5, 0.] * 4,
        #                     device=self._device)

        robot_lb = to_torch(
            [-0.1, -0.1, 0., -3.14, -3.14, -3.14, -1., -1., -1., -3.14, -3.14, -3.14], device=self._device)
        robot_ub = to_torch([0.1, 0.1, 0.6, 3.14, 3.14, 3.14, 1., 1., 1., 3.14, 3.14, 3.14],
                            device=self._device)

        task_lb = to_torch([-2., -2., -1., -1., -1.], device=self._device)
        task_ub = to_torch([2., 2., 1., 1., 1.], device=self._device)
        # self._observation_lb = torch.concatenate((task_lb, robot_lb))
        # self._observation_ub = torch.concatenate((task_ub, robot_ub))
        self._observation_lb = robot_lb
        self._observation_ub = robot_ub
        if self._config.get("observe_heights", False):
            num_heightpoints = len(self._config.measured_points_x) * len(
                self._config.measured_points_y)
            self._observation_lb = torch.concatenate(
                (self._observation_lb, torch.zeros(num_heightpoints, device=self._device) - 3))
            self._observation_ub = torch.concatenate(
                (self._observation_ub, torch.zeros(num_heightpoints, device=self._device) + 3))
        self._action_lb = to_torch(self._config.action_lb, device=self._device) * 5
        self._action_ub = to_torch(self._config.action_ub, device=self._device) * 5

    def _prepare_rewards(self):
        self._reward_names, self._reward_fns, self._reward_scales = [], [], []
        self._episode_sums = dict()
        for name, scale in self._config.rewards:
            self._reward_names.append(name)
            self._reward_fns.append(getattr(self._rewards, name + '_reward'))
            self._reward_scales.append(scale)
            self._episode_sums[name] = torch.zeros(self._num_envs,
                                                   device=self._device)

        self._terminal_reward_names, self._terminal_reward_fns, self._terminal_reward_scales = [], [], []
        for name, scale in self._config.terminal_rewards:
            self._terminal_reward_names.append(name)
            self._terminal_reward_fns.append(getattr(self._rewards, name + '_reward'))
            self._terminal_reward_scales.append(scale)
            self._episode_sums[name] = torch.zeros(self._num_envs,
                                                   device=self._device)

    def reset(self) -> torch.Tensor:
        return self.reset_idx(torch.arange(self._num_envs, device=self._device))

    def _split_action(self, action):
        gait_action = None
        if self._config.get('include_gait_action', False):
            gait_action = action[:, :1]
            action = action[:, 1:]

        foot_action = None
        if self._config.get('include_foot_action', False):
            if self._config.get('mirror_foot_action', False):
                foot_action = action[:, -6:].reshape(
                    (-1, 2, 3))  # + self._robot.hip_offset
                foot_action = torch.stack([
                    foot_action[:, 0],
                    foot_action[:, 0],
                    foot_action[:, 1],
                    foot_action[:, 1],
                ], dim=1)
                action = action[:, :-6]
            else:
                foot_action = action[:, -12:].reshape(
                    (-1, 4, 3))  # + self._robot.hip_offset
                action = action[:, :-12]

        com_action = action
        return gait_action, com_action, foot_action

    def reset_idx(self, env_ids) -> torch.Tensor:
        # Aggregate rewards
        self._extras["time_outs"] = self._episode_terminated()
        if env_ids.shape[0] > 0:
            self._extras["episode"] = {}
            print(f"self._gait_generator.true_phase[env_ids]: {self._gait_generator.true_phase[env_ids]}")
            # time.sleep(123)
            self._extras["episode"]["cycle_count"] = torch.mean(
                self._gait_generator.true_phase[env_ids]) / (2 * torch.pi)

            self._obs_buf = self._get_observations()
            self._privileged_obs_buf = self._get_privileged_observations()

            for reward_name in self._episode_sums.keys():
                if reward_name in self._reward_names:
                    if self._config.get('normalize_reward_by_phase', False):
                        self._extras["episode"]["reward_{}".format(
                            reward_name)] = torch.mean(
                            self._episode_sums[reward_name][env_ids] /
                            (self._gait_generator.true_phase[env_ids] / (2 * torch.pi)))
                    else:
                        # Normalize by time
                        self._extras["episode"]["reward_{}".format(
                            reward_name)] = torch.mean(
                            self._episode_sums[reward_name][env_ids] /
                            (self._steps_count[env_ids] * (self._config.env_dt)))

                if reward_name in self._terminal_reward_names:
                    self._extras["episode"]["reward_{}".format(reward_name)] = torch.mean(
                        self._episode_sums[reward_name][env_ids] /
                        (self._cycle_count[env_ids].clip(min=1)))

                self._episode_sums[reward_name][env_ids] = 0

            self._steps_count[env_ids] = 0
            self._cycle_count[env_ids] = 0
            self._init_yaw[env_ids] = self._robot.base_orientation_rpy[env_ids, 2]
            self._robot.reset_idx(env_ids)
            self._swing_leg_controller.reset_idx(env_ids)
            self._gait_generator.reset_idx(env_ids)
            # self._resample_command(env_ids)

            # self.load_plane_asset()

        return self._obs_buf, self._privileged_obs_buf

    def step(self, action: torch.Tensor):
        self._cnt += 1
        # time.sleep(1)
        # print(f"action is: {action}")
        self._last_obs_buf = torch.clone(self._obs_buf)

        self._last_action = torch.clone(action)
        print(f"action before clip is: {action}")
        print(f"self._action_lb: {self._action_lb}")
        print(f"self._action_ub: {self._action_ub}")
        action = torch.clip(action, self._action_lb, self._action_ub)
        print(f"step action is: {action}")
        # time.sleep(1)
        # action = torch.zeros_like(action)
        sum_reward = torch.zeros(self._num_envs, device=self._device)
        dones = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._steps_count += 1
        logs = []

        zero = torch.zeros(self._num_envs, device=self._device)
        # gait_action, com_action, foot_action = self._split_action(action)
        com_action = action
        # time.sleep(123)
        start = time.time()
        # self._config.env_dt = 0.002

        for step in range(max(int(self._config.env_dt / self._robot.control_timestep), 1)):
            # print(f"config.env_dt: {self._config.env_dt}")
            # print(f"self._robot.control_timestep: {self._robot.control_timestep}")
            self._gait_generator.update()
            self._swing_leg_controller.update()
            if self._use_real_robot:
                self._robot.state_estimator.update_foot_contact(
                    self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error
                self._robot.update_desired_foot_contact(
                    self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error

            # if gait_action is not None:
            #     self._gait_generator.stepping_frequency = gait_action[:, 0]

            # CoM pose action
            self._torque_optimizer.desired_base_position = torch.stack(
                (zero, zero,
                 # com_action[:, 0]),
                 torch.full((self._num_envs,), self.desired_com_height).to(self._device)),
                dim=1)
            # self._torque_optimizer.desired_linear_velocity = torch.stack(
            #     (com_action[:, 1], com_action[:, 2] * 0, com_action[:, 3]), dim=1)
            # self._torque_optimizer.desired_base_orientation_rpy = torch.stack(
            #     (com_action[:, 4] * 0, com_action[:, 5],
            #      self._robot.base_orientation_rpy[:, 2]),
            #     dim=1)
            self._torque_optimizer.desired_linear_velocity = torch.stack(
                (torch.full((self._num_envs,), self.desired_vx).to(self._device), zero, zero), dim=1)
            self._torque_optimizer.desired_base_orientation_rpy = torch.stack(
                (com_action[:, 4] * 0, com_action[:, 5] * 0, zero), dim=1)

            # self._torque_optimizer.desired_angular_velocity = torch.stack(
            #     (zero, zero, torch.full((self._num_envs,), 0.5).to(self._device)), dim=1)

            print(f"self._torque_optimizer.desired_angular_velocity: {self._torque_optimizer.desired_angular_velocity}")
            # time.sleep(123)
            # if self._config.get('use_yaw_feedback', False):
            #     yaw_err = (self._init_yaw - self._robot.base_orientation_rpy[:, 2])
            #     yaw_err = torch.remainder(yaw_err + 3 * torch.pi,
            #                               2 * torch.pi) - torch.pi
            #     desired_yaw_rate = 1 * yaw_err
            #     self._torque_optimizer.desired_angular_velocity = torch.stack(
            #         (zero, com_action[:, 6], desired_yaw_rate), dim=1)
            # else:
            #     self._torque_optimizer.desired_angular_velocity = torch.stack(
            #         (zero, com_action[:, 6], com_action[:, 7] * 0), dim=1)

            # if self._config.get('use_yaw_feedback', False):
            #     yaw_err = (self._init_yaw - self._robot.base_orientation_rpy[:, 2])
            #     yaw_err = torch.remainder(yaw_err + 3 * torch.pi,
            #                               2 * torch.pi) - torch.pi
            #     desired_yaw_rate = 1 * yaw_err
            #     self._torque_optimizer.desired_angular_velocity = torch.stack(
            #         (zero, zero, zero), dim=1)
            # else:
            #     self._torque_optimizer.desired_angular_velocity = torch.stack(
            #         (zero, zero, com_action[:, 7] * 0), dim=1)

            desired_foot_positions = self._swing_leg_controller.desired_foot_positions
            # if foot_action is not None:
            #     base_yaw = self._robot.base_orientation_rpy[:, 2]
            #     cos_yaw = torch.cos(base_yaw)[:, None]
            #     sin_yaw = torch.sin(base_yaw)[:, None]
            #     foot_action[:, :, 0] *= -1  # DEBUG HACK
            #     foot_action_world = torch.clone(foot_action)
            #     foot_action_world[:, :, 0] = (cos_yaw * foot_action[:, :, 0] -
            #                                   sin_yaw * foot_action[:, :, 1])
            #     foot_action_world[:, :, 1] = (sin_yaw * foot_action[:, :, 0] +
            #                                   cos_yaw * foot_action[:, :, 1])
            #     desired_foot_positions += foot_action_world
            motor_action, self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = self._torque_optimizer.get_action(
                self._gait_generator.desired_contact_state,
                swing_foot_position=desired_foot_positions,
                residual_acc=None)
            hp_action = action * 3 + self._desired_acc

            self._robot.energy_2d = self.ha_teacher.update(
                np.asarray(self._torque_optimizer.tracking_error.cpu()))

            ha_action, dwell_flag = self.ha_teacher.get_action()
            ha_action = to_torch(ha_action, device=self._device)

            # Use Normal Kp Kd
            # ha_action = self._desired_acc.squeeze()

            print(f"hp_action: {hp_action}")
            print(f"ha_action: {ha_action}")
            print(f"self._torque_optimizer.tracking_error: {self._torque_optimizer.tracking_error}")
            terminal_stance_ddq, action_mode = self.coordinator.get_terminal_action(hp_action=hp_action,
                                                                                    ha_action=ha_action,
                                                                                    plant_state=np.asarray(
                                                                                        self._torque_optimizer.tracking_error.cpu()),
                                                                                    dwell_flag=dwell_flag,
                                                                                    epsilon=self.ha_teacher.epsilon)
            # time.sleep(0.1)
            # terminal_stance_ddq = torch.tile(torch.tensor(terminal_stance_ddq, dtype=torch.float32),
            #                                  dims=(self._num_envs, 1))
            # terminal_stance_ddq = hp_action
            terminal_stance_ddq = to_torch(terminal_stance_ddq, device=self._device)
            print(f"terminal_stance_ddq: {terminal_stance_ddq}")
            print(f"action_mode: {action_mode}")
            # time.sleep(123)

            # motor_action, self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = self._torque_optimizer.get_action(
            #     self._gait_generator.desired_contact_state,
            #     swing_foot_position=desired_foot_positions,
            #     residual_acc=action)

            motor_action, self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = self._torque_optimizer.get_safe_action(
                self._gait_generator.desired_contact_state,
                swing_foot_position=desired_foot_positions,
                safe_acc=terminal_stance_ddq)

            # print(f"desired_acc: {self._desired_acc}")
            # print(f"solved_acc: {self._solved_acc}")
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
                     num_clips=self._num_clips,
                     foot_contact_state=self._gait_generator.desired_contact_state,
                     foot_contact_force=self._robot.foot_contact_forces,
                     desired_swing_foot_position=desired_foot_positions,
                     desired_acc_body_frame=self._desired_acc,
                     desired_vx=self.desired_vx,
                     desired_com_height=self.desired_com_height,
                     ha_action=ha_action,
                     hp_action=hp_action,
                     action_mode=action_mode,
                     acc_min=to_torch([-10, -10, -10, -20, -20, -20], device=self._device),
                     acc_max=to_torch([10, 10, 10, 20, 20, 20], device=self._device),
                     energy=to_torch(self._robot.energy_2d, device=self._device),
                     solved_acc_body_frame=self._solved_acc,
                     foot_positions_in_base_frame=self._robot.
                     foot_positions_in_base_frame,
                     env_action=action,
                     env_obs=torch.clone(self._obs_buf)))
            if self._use_real_robot:
                logs[-1]["base_acc"] = np.array(
                    self._robot.raw_state.imu.accelerometer)  # pytype: disable=attribute-error

            s_desired = torch.cat((
                self._torque_optimizer.desired_base_position,
                self._torque_optimizer.desired_base_orientation_rpy,
                self._torque_optimizer.desired_linear_velocity,
                self._torque_optimizer.desired_angular_velocity),
                dim=-1)
            print(f"s_desired: {s_desired}")
            s_old = torch.cat((
                self._robot.base_position,
                self._robot.base_orientation_rpy,
                self._robot.base_velocity_body_frame,
                self._robot.base_angular_velocity_body_frame,
            ), dim=-1)
            print(f"s_old: {s_old}")

            s = (s_old - s_desired)

            self._robot.step(motor_action)

            self._obs_buf = self._get_observations()
            self._privileged_obs_buf = self.get_privileged_observations()
            # rewards = self.get_reward()

            s_desired_next = torch.cat((
                self._torque_optimizer.desired_base_position,
                self._torque_optimizer.desired_base_orientation_rpy,
                self._torque_optimizer.desired_linear_velocity,
                self._torque_optimizer.desired_angular_velocity),
                dim=-1)

            s_old_next = torch.cat((
                self._robot.base_position,
                self._robot.base_orientation_rpy,
                self._robot.base_velocity_body_frame,
                self._robot.base_angular_velocity_body_frame,
            ), dim=-1)
            s_next = (s_old_next - s_desired_next)
            rewards = self.get_reward_ly(s=s, s_next=s_next)

            dones = torch.logical_or(dones, self._is_done())
            # print(f"rewards: {rewards.shape}")
            # print(f"dones: {dones.shape}")
            # print(f"sum_reward: {sum_reward.shape}")
            sum_reward += rewards * torch.logical_not(dones)

            push_rng = np.random.default_rng(seed=42)

            def _push_robots():
                """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
                """
                # time.sleep(2)
                # self.arrow_plot()

                from isaacgym import gymtorch
                # max_vel = self.cfg.domain_rand.max_push_vel_xy
                max_vel = 1
                # torch.manual_seed(30)
                # self.robot._root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                #                                                    device=self.device)  # lin vel x/y

                curr_vx = self.robot._root_states[:self._num_envs, 7]
                curr_vy = self.robot._root_states[:self._num_envs, 8]
                curr_vz = self.robot._root_states[:self._num_envs, 9]
                print(f"self.robot._root_states: {self.robot._root_states}")
                # time.sleep(123)

                sgn = -1 if dual_push_sequence[self._push_cnt] % 2 == 0 else 1
                # if self._push_cnt == 0:
                #     self._push_magnitude *= sgn
                # elif sgn == -1:
                #     self._push_magnitude = -1.1
                # elif sgn == 1:
                #     self._push_magnitude = 1.9
                # self._push_magnitude = push_magnitude_list[self._push_cnt]
                # Original static push
                # delta_x = -0.25
                # delta_y = 0.62 * self._push_magnitude
                # delta_z = -0.72

                delta_x = push_delta_vel_list_x[self._push_cnt]
                delta_y = push_delta_vel_list_y[self._push_cnt]
                delta_z = push_delta_vel_list_z[self._push_cnt]

                # Random push from uniform distribution
                # new_seed = push_sequence[self._push_cnt]
                # print(f"new_seed: {new_seed}")
                # np.random.seed(new_seed)
                # sgn = -1 if new_seed % 2 == 0 else 1
                #
                # delta_x = np.random.uniform(low=0., high=0.2) * sgn
                # delta_y = np.random.uniform(low=0., high=0.6) * sgn
                # delta_z = np.random.uniform(low=-0.8, high=-0.7)
                print(f"delta_x: {delta_x}")
                print(f"delta_y: {delta_y}")
                print(f"delta_z: {delta_z}")
                # time.sleep(2)

                vel_after_push_x = curr_vx + delta_x
                vel_after_push_y = curr_vy + delta_y
                vel_after_push_z = curr_vz + delta_z

                print(f"vel_after_push_x: {vel_after_push_x}")
                print(f"vel_after_push_y: {vel_after_push_y}")
                print(f"vel_after_push_z: {vel_after_push_z}")

                # Turn on the push indicator for viewing
                self.draw_push_indicator(target_pos=[delta_x, delta_y, delta_z])

                # time.sleep(1)
                # print(f"delta_x: {curr_x}")
                self.robot._root_states[:self._num_envs, 7] = torch.full((self.num_envs, 1), vel_after_push_x.item())
                self.robot._root_states[:self._num_envs, 8] = torch.full((self.num_envs, 1), vel_after_push_y.item())
                # print(f"FFFFFFFFFFFFFFFFFFFFFFFFFFFF: {torch.full((self.num_envs, 1), .2)}")
                self.robot._root_states[:self._num_envs, 9] = torch.full((self.num_envs, 1), vel_after_push_z.item())
                # self._gym.set_actor_root_state_tensor(self._sim, gymtorch.unwrap_tensor(self.robot._root_states))

                actor_count = self._gym.get_env_count(self._sim)
                # self.robot._root_states = self.robot._root_states.repeat(7, 1)
                indices = to_torch([i for i in range(self._num_envs)], dtype=torch.int32, device=self._device)
                indices_tensor = gymtorch.unwrap_tensor(indices)
                self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                              gymtorch.unwrap_tensor(self.robot._root_states),
                                                              indices_tensor,
                                                              1)
                self._push_cnt += 1
                # self._gym.set_dof_state_tensor(self._sim, gymtorch.unwrap_tensor(self.robot._root_states))

            # torch.manual_seed(42)
            random_interval = torch.randint(100, 200, (1,), dtype=torch.int64, device=self.device)
            # if self._cnt % random_interval == 0:

            if self._indicator_flag:
                if self._indicator_cnt < 30:
                    self._indicator_cnt += 1
                else:
                    self._gym.clear_lines(self._viewer)
                    self._indicator_cnt = 0
                    self._indicator_flag = False

            # Origin push method (static interval)
            # if self._cnt > 298 and (self._cnt + 1) % 150 == 0:
            #     print(f"cnt is: {self._cnt}, pushing the robot now")
            #
            #     _push_robots()
            #     self._indicator_flag = True
            #     # time.sleep(1)
            #
            #     # self._gym.add_lines(self.robot._envs[0], None, 0, [])

            # New push method (Dynamic method)
            if self._cnt > 298 and self._cnt == push_interval[self._push_cnt]:
                print(f"cnt is: {self._cnt}, pushing the robot now")

                # _push_robots()
                self._indicator_flag = True
                # time.sleep(1)

                # self._gym.add_lines(self.robot._envs[0], None, 0, [])

            # print(f"Time: {self._robot.time_since_reset}")
            # print(f"Gait: {gait_action}")
            # print(f"Foot: {foot_action}")
            # print(f"Phase: {self._obs_buf[:, 3]}")
            # print(f"Desired contact: {self._gait_generator.desired_contact_state}")
            # print(f"Desired Position: {self._torque_optimizer.desired_base_position}")
            # print(f"Current Position: {self._robot.base_position}")
            # print(
            #     f"Desired Velocity: {self._torque_optimizer.desired_linear_velocity}")
            # print(f"Current Velocity: {self._robot.base_velocity_world_frame}")
            # print(
            #     f"Desired RPY: {self._torque_optimizer.desired_base_orientation_rpy}")
            # print(f"Current RPY: {self._robot.base_orientation_rpy}")
            # print(
            #     f"Desired Angular Vel: {self._torque_optimizer.desired_angular_velocity}"
            # )
            # print(
            #     f"Current Angular vel: {self._robot.base_angular_velocity_body_frame}")
            # print(f"Desired Acc: {self._desired_acc}")
            # print(f"Solved Acc: {self._solved_acc}")
            # ans = input("Any Key...")
            # if ans in ["Y", "y"]:
            #   import pdb
            #   pdb.set_trace()
            self._extras["logs"] = logs
            # Resample commands
            new_cycle_count = (self._gait_generator.true_phase / (2 * torch.pi)).long()
            finished_cycle = new_cycle_count > self._cycle_count
            env_ids_to_resample = finished_cycle.nonzero(as_tuple=False).flatten()
            self._cycle_count = new_cycle_count

            is_terminal = torch.logical_or(finished_cycle, dones)
            if is_terminal.any():
                print(f"terminal_reward is: {self.get_terminal_reward(is_terminal, dones)}")
                # sum_reward += self.get_terminal_reward(is_terminal, dones)
            # print(self.get_terminal_reward(is_terminal))
            # import pdb
            # pdb.set_trace()
            # self._resample_command(env_ids_to_resample)
            if not self._use_real_robot:
                pass
                # self.reset_idx(dones.nonzero(as_tuple=False).flatten())

            # if dones.any():
            #   import pdb
            #   pdb.set_trace()
            # print(f"sum_reward: {sum_reward}")
            # print(f"sum_reward: {sum_reward.shape}")
            if self._show_gui:
                self._robot.render()

        end = time.time()
        print(
            f"*************************************** step duration: {end - start} ***************************************")
        return self._obs_buf, self._privileged_obs_buf, sum_reward, dones, self._extras

    def _get_observations(self):

        # robot_obs = torch.concatenate(
        #     (
        #         self._robot.base_position[:, 2:],  # Base height
        #         self._robot.base_orientation_rpy[:, 0:1] * 0,  # Base roll
        #         self._robot.base_orientation_rpy[:, 1:2],  # Base Pitch
        #         self._robot.base_velocity_body_frame[:, 0:1],
        #         self._robot.base_velocity_body_frame[:, 1:2] * 0,
        #         self._robot.base_velocity_body_frame[:, 2:3],  # Base velocity (z)
        #         self._robot.base_angular_velocity_body_frame[:, 0:1] * 0,
        #         self._robot.base_angular_velocity_body_frame[:, 1:2],
        #         self._robot.base_angular_velocity_body_frame[:, 2:3],  # Base yaw rate
        #         # self._robot.motor_positions,
        #         # self._robot.motor_velocities,
        #         # self._robot.foot_positions_in_base_frame.reshape((self._num_envs, 12)),),
        #     ),
        #     dim=1)

        s_desired_next = torch.cat((
            self._torque_optimizer.desired_base_position,
            self._torque_optimizer.desired_base_orientation_rpy,
            self._torque_optimizer.desired_linear_velocity,
            self._torque_optimizer.desired_angular_velocity),
            dim=-1)

        s_old_next = torch.cat((
            self._robot.base_position,
            self._robot.base_orientation_rpy,
            self._robot.base_velocity_body_frame,
            self._robot.base_angular_velocity_body_frame,
        ), dim=-1)

        robot_obs = (s_old_next - s_desired_next)
        print(f"s_desired_next: {s_desired_next}")
        print(f"s_old_next: {s_old_next}")
        print(f"robot_obs: {robot_obs}")
        print(f"robot_obs: {robot_obs.shape}")

        # robot_obs = torch.concatenate(
        #     (
        #         self._robot.base_position[:, :],  # Base height
        #         self._robot.base_orientation_rpy[:, :] ,  # Base roll
        #         self._robot.base_velocity_body_frame[:, :],  # Base velocity (z)
        #         self._robot.base_angular_velocity_body_frame[:, :],  # Base yaw rate
        #         # self._robot.motor_positions,
        #         # self._robot.motor_velocities,
        #         # self._robot.foot_positions_in_base_frame.reshape((self._num_envs, 12)),),
        #     ),
        #     dim=1)

        # print(f"robot_obs: {robot_obs}")
        # print(f"robot_obs: {robot_obs.shape}")
        # print(f"robot_obs: {phase_obs}")
        # print(f"robot_obs: {phase_obs.shape}")
        # print(f"distance_to_goal_local: {distance_to_goal_local}")
        # print(f"distance_to_goal_local: {distance_to_goal_local.shape}")
        # time.sleep(123)
        # obs = torch.concatenate((distance_to_goal_local, phase_obs, robot_obs),dim=1)
        obs = robot_obs
        if self._config.get("observation_noise",
                            None) is not None and (not self._use_real_robot):
            obs += torch.randn_like(obs) * self._config.observation_noise
        return obs

    def get_observations(self):
        return self._obs_buf

    def _get_privileged_observations(self):
        return None

    def get_privileged_observations(self):
        return self._privileged_obs_buf

    def get_reward(self):
        sum_reward = torch.zeros(self._num_envs, device=self._device)
        for idx in range(len(self._reward_names)):
            reward_name = self._reward_names[idx]
            reward_fn = self._reward_fns[idx]
            reward_scale = self._reward_scales[idx]
            reward_item = reward_scale * reward_fn()
            if self._config.get('normalize_reward_by_phase', False):
                reward_item *= self._gait_generator.stepping_frequency
            self._episode_sums[reward_name] += reward_item
            sum_reward += reward_item

        if self._config.clip_negative_reward:
            sum_reward = torch.clip(sum_reward, min=0)
        return sum_reward

    def get_reward_ly(self, s, s_next):
        # sum_reward = torch.zeros(self._num_envs, device=self._device)

        MATRIX_P = torch.tensor([[140.6434, 0, 0, 0, 0, 0, 5.3276, 0, 0, 0],
                                 [0, 134.7596, 0, 0, 0, 0, 0, 6.6219, 0, 0],
                                 [0, 0, 134.7596, 0, 0, 0, 0, 0, 6.622, 0],
                                 [0, 0, 0, 49.641, 0, 0, 0, 0, 0, 6.8662],
                                 [0, 0, 0, 0, 11.1111, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 3.3058, 0, 0, 0, 0],
                                 [5.3276, 0, 0, 0, 0, 0, 3.6008, 0, 0, 0],
                                 [0, 6.6219, 0, 0, 0, 0, 0, 3.6394, 0, 0],
                                 [0, 0, 6.622, 0, 0, 0, 0, 0, 3.6394, 0],
                                 [0, 0, 0, 6.8662, 0, 0, 0, 0, 0, 4.3232]], device=self._device)
        s_new = s[:, 2:]
        s_next_new = s_next[:, 2:]
        # print(f"s: {s.shape}")
        # print(f"s_new: {s_new.shape}")
        # ly_reward_curr = s_new.T @ MATRIX_P @ s_new
        ST = torch.matmul(s_new, MATRIX_P)
        ly_reward_curr = torch.sum(ST * s_new, dim=1, keepdim=True)

        # ly_reward_next = s_next_new.T @ MATRIX_P @ s_next_new
        ST = torch.matmul(s_next_new, MATRIX_P)
        ly_reward_next = torch.sum(ST * s_next_new, dim=1, keepdim=True)

        sum_reward = ly_reward_curr - ly_reward_next  # multiply scaler to decrease
        # print(f"ly_reward_curr: {ly_reward_curr.shape}")
        # print(f"ly_reward_next: {ly_reward_next.shape}")
        # print(f"sum_reward: {sum_reward.shape}")
        # print(f"sum_reward: {sum_reward}")
        # sum_reward = torch.tensor(reward, device=self._device)

        # for idx in range(len(self._reward_names)):
        #     reward_name = self._reward_names[idx]
        #     reward_fn = self._reward_fns[idx]
        #     reward_scale = self._reward_scales[idx]
        #     reward_item = reward_scale * reward_fn()
        #     if self._config.get('normalize_reward_by_phase', False):
        #         reward_item *= self._gait_generator.stepping_frequency
        #     self._episode_sums[reward_name] += reward_item
        #     sum_reward += reward_item
        #
        # if self._config.clip_negative_reward:
        #     sum_reward = torch.clip(sum_reward, min=0)
        return sum_reward.squeeze(dim=-1)

    def get_terminal_reward(self, is_terminal, dones):
        early_term = torch.logical_and(
            dones, torch.logical_not(self._episode_terminated()))
        coef = torch.where(early_term, self._gait_generator.cycle_progress,
                           torch.ones_like(early_term))

        sum_reward = torch.zeros(self._num_envs, device=self._device)
        for idx in range(len(self._terminal_reward_names)):
            reward_name = self._terminal_reward_names[idx]
            reward_fn = self._terminal_reward_fns[idx]
            reward_scale = self._terminal_reward_scales[idx]
            reward_item = reward_scale * reward_fn() * is_terminal * coef
            self._episode_sums[reward_name] += reward_item
            sum_reward += reward_item

        if self._config.clip_negative_terminal_reward:
            sum_reward = torch.clip(sum_reward, min=0)
        return sum_reward

    def _episode_terminated(self):
        timeout = (self._steps_count >= self._episode_length)
        cycles_finished = (self._gait_generator.true_phase /
                           (2 * torch.pi)) > self._config.get('max_jumps', 1)
        return torch.logical_or(timeout, cycles_finished)

    def _is_done(self):
        is_unsafe = torch.logical_or(
            self._robot.projected_gravity[:, 2] < 0.5,
            self._robot.base_position[:, 2] < torch.tensor(self._config.get('terminate_on_height', 0.15),
                                                           dtype=torch.bool))
        if self._config.get('terminate_on_body_contact', False):
            is_unsafe = torch.logical_or(is_unsafe, self._robot.has_body_contact)

        if self._config.get('terminate_on_limb_contact', False):
            limb_contact = torch.logical_or(self._robot.calf_contacts,
                                            self._robot.thigh_contacts)
            limb_contact = torch.sum(limb_contact, dim=1)
            is_unsafe = torch.logical_or(is_unsafe, limb_contact > 0)

        # print(self._robot.base_position[:, 2])
        # input("Any Key...")
        # if is_unsafe.any():
        #   import pdb
        #   pdb.set_trace()
        return torch.logical_or(self._episode_terminated(), is_unsafe)

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
    def desired_landing_position(self):
        return self._desired_landing_position

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
        return self._steps_count

    @episode_length_buf.setter
    def episode_length_buf(self, new_length: torch.Tensor):
        self._steps_count = to_torch(new_length, device=self._device)
        self._gait_generator._current_phase += 2 * torch.pi * (new_length / self.max_episode_length * self._config.get(
            'max_jumps', 1) + 1)[:, None]
        self._cycle_count = (self._gait_generator.true_phase /
                             (2 * torch.pi)).long()

    @property
    def device(self):
        return self._device

    def draw_push_indicator(self, target_pos=[1., 0., 0.]):
        sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.01, 50, 50, None, color=(1, 0., 0.))
        pose_robot = self.robot._root_states[:self._num_envs, :3].squeeze(dim=0).cpu().numpy()
        print(f"pose_robot: {pose_robot}")
        self.target_pos_rel = to_torch([target_pos], device=self._device)
        for i in range(5):
            norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            print(f"norm: {norm}")
            print(f"target_vec_norm: {target_vec_norm}")
            # pose_arrow = pose_robot[:3] + 0.1 * (i + 3) * target_vec_norm[:self._num_envs, :3].cpu().numpy()

            xy = pose_robot[:2] + 0.08 * (i + 3) * target_vec_norm[:self._num_envs, :2].cpu().numpy()
            z = pose_robot[2] + 0.03 * (i + 3) * target_vec_norm[:self._num_envs, 2].cpu().numpy()
            print(f"xy: {xy}")
            print(f"xy: {z}")
            pose_arrow = np.hstack((xy.squeeze(), z))
            # pose_arrow = pose_arrow.squeeze()
            print(f"pose_arrow: {pose_arrow}")
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            print(f"pose: {pose}")
            gymutil.draw_lines(sphere_geom_arrow, self._gym, self._viewer, self.robot._envs[0], pose)

        sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
        # for i in range(5):
        #     norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        #     target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        #     pose_arrow = pose_robot[:2] + 0.2 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
        #     pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
        #     gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
