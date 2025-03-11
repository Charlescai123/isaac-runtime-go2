"""Configuration for Go2 Trot Env"""
import time

from ml_collections import ConfigDict
from src.configs.training import ppo, ddpg
from src.envs import go2_wild_env
import torch
import numpy as np


def get_env_config():
    """Config for Environment"""

    config = ConfigDict()

    # Observation: [dis2goal, yaw_deviation, height, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    config.observation_lb = np.array([0., 0., 0., -3.14, -3.14, -3.14, -1., -1., -1., -3.14, -3.14, -3.14])
    config.observation_ub = np.array([10., 3.14, 0.6, 3.14, 3.14, 3.14, 1., 1., 1., 3.14, 3.14, 3.14])

    # Action: [acc_vx, acc_vy, acc_vz, acc_wx, acc_wy, acc_wz]
    config.action_lb = np.array([-10., -10., -10., -20., -20., -20.])
    config.action_ub = np.array([10., 10., 10., 20., 20., 20.])

    # DRL-based gamma (0 < gamma < 1)
    config.gamma = 0.35
    config.obs_dim = config.observation_lb.shape[0]
    config.act_dim = config.action_lb.shape[0]
    config.episode_length_s = 120.
    config.env_dt = 0.01
    # config.env_dt = 0.002
    config.motor_strength_ratios = 1.
    config.motor_torque_delay_steps = 5
    config.use_yaw_feedback = False

    # Safety subset (height, roll, pitch, yaw, vx, vy, vz, wx, wy, wz)
    safety_subset = [0.12, 0.35, 0.35, np.inf, 0.35, np.inf, np.inf, np.inf, np.inf, 1.]

    # HA-Teacher
    ha_teacher_config = ConfigDict()
    ha_teacher_config.chi = 0.2
    ha_teacher_config.tau = 10
    ha_teacher_config.enable = False
    ha_teacher_config.correct = True
    ha_teacher_config.epsilon = 1
    ha_teacher_config.cvxpy_solver = "solver"
    ha_teacher_config.safety_subset = safety_subset
    config.ha_teacher = ha_teacher_config

    # Gait config
    gait_config = ConfigDict()
    gait_config.stepping_frequency = 2
    gait_config.initial_offset = np.array([0., 0.5, 0.5, 0.], dtype=np.float32) * (2 * np.pi)
    gait_config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    config.gait = gait_config

    # Stance controller
    config.base_position_kp = np.array([0., 0., 25.]) * 2
    config.base_position_kd = np.array([5., 5., 5.]) * 2
    config.base_orientation_kp = np.array([25., 25., 0.]) * 2
    config.base_orientation_kd = np.array([5., 5., 5.]) * 2
    config.qp_foot_friction_coef = 0.7
    config.qp_weight_ddq = np.diag([1., 1., 10., 10., 10., 1.])
    config.qp_body_inertia = np.array([[0.1585, 0.0001, -0.0155],
                                       [0.0001, 0.4686, 0.],
                                       [-0.0155, 0., 0.5245]]),
    config.use_full_qp = False
    config.clip_grf_in_sim = True
    config.foot_friction = 0.7  # 0.7
    config.desired_vx = 0.4  # vx
    config.desired_pz = 0.3  # height
    config.desired_wz = 0.  # wz
    config.clip_wz = [-0.7, 0.7]  # clip desired_wz

    # Swing controller
    config.swing_foot_height = 0.13
    config.swing_foot_landing_clearance = 0.02

    # Termination condition
    config.terminate_on_destination_reach = True
    config.terminate_on_body_contact = False
    config.terminate_on_dense_body_contact = True
    config.terminate_on_limb_contact = False
    config.terminate_on_yaw_deviation = False
    config.terminate_on_out_of_terrain = True
    config.terminate_on_timeout = True
    config.terminate_on_height = 0.15
    config.terrain_region = [[42, 62], [-7, 7]]
    config.yaw_deviation_threshold = np.pi * 0.75
    config.use_penetrating_contact = False

    # Reward
    reward_config = ConfigDict()
    reward_config.scales = {
        # 'upright': 0.02,
        # 'contact_consistency': 0.008,
        # 'foot_slipping': 0.032,
        # 'foot_clearance': 0.008,
        # 'out_of_bound_action': 0.01,
        # 'knee_contact': 5,
        # 'stepping_freq': 0.008,
        # 'com_distance_to_goal_squared': 0.016,
        # 'jerky_action': -1,
        # 'alive': 10,
        'fall_down': 10000,
        # 'forward_speed': 0.1,
        # 'lin_vel_z': -2,
        'body_contact': 100,
        'energy_consumption': 0.05,
        'lin_vel_tracking': 5,
        'ang_vel_tracking': 5,
        'orientation_tracking': 80,
        'height_tracking': 50,
        'lyapunov': 100,
        # 'reach_time': 10,
        'distance_to_wp': 50,      # Distance to waypoint
        'reach_wp': 100,            # Reach waypoint
        'reach_goal': 500,         # Reach destination
        # 'com_height': 0.01,
    }
    reward_config.only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
    reward_config.tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
    reward_config.soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
    reward_config.soft_dof_vel_limit = 1.
    reward_config.soft_torque_limit = 1.
    reward_config.base_height_target = 1.
    reward_config.max_contact_force = 100.  # forces above this value are penalized
    config.reward = reward_config

    config.clip_negative_reward = False
    config.normalize_reward_by_phase = True

    config.terminal_rewards = []
    config.clip_negative_terminal_reward = False
    return config


def get_config():
    """Main entrance for the parsing the config"""
    config = ConfigDict()
    # config.training = ppo.get_training_config()  # Use PPO
    config.training = ddpg.get_training_config()  # Use DDPG
    config.env_class = go2_wild_env.Go2WildExploreEnv
    config.environment = get_env_config()
    return config
