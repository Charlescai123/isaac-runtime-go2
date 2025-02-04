"""Configuration for Go2 Trot Env"""
from ml_collections import ConfigDict
from src.configs.training import ppo, ddpg
from src.envs import go2_trot_env
import torch
import numpy as np


def get_env_config():
    """Config for Environment"""

    config = ConfigDict()

    # HA-Teacher
    ha_teacher_config = ConfigDict()
    ha_teacher_config.chi = 0.15
    ha_teacher_config.tau = 100
    ha_teacher_config.enable = False
    ha_teacher_config.correct = True
    ha_teacher_config.epsilon = 1
    ha_teacher_config.cvxpy_solver = "solver"
    config.ha_teacher = ha_teacher_config

    # Gait config
    gait_config = ConfigDict()
    gait_config.stepping_frequency = 2
    gait_config.initial_offset = np.array([0., 0.5, 0.5, 0.], dtype=np.float32) * (2 * np.pi)
    gait_config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    config.gait = gait_config

    # Fully Flexible
    config.observation_lb = np.array([-0.01, -0.01, 0., -3.14, -3.14, -3.14, -1., -1., -1., -3.14, -3.14, -3.14])
    config.observation_ub = np.array([0.01, 0.01, 0.6, 3.14, 3.14, 3.14, 1., 1., 1., 3.14, 3.14, 3.14])
    config.action_lb = np.array([-10., -10., -10., -20., -20., -20.])
    config.action_ub = np.array([10., 10., 10., 20., 20., 20.])

    config.episode_length_s = 20.
    config.max_jumps = 10.
    config.env_dt = 0.01
    # config.env_dt = 0.002
    config.motor_strength_ratios = 1.
    config.motor_torque_delay_steps = 5
    config.use_yaw_feedback = False

    # Stance controller
    config.base_position_kp = np.array([0., 0., 25]) * 2
    config.base_position_kd = np.array([5., 5., 5.]) * 2
    config.base_orientation_kp = np.array([25., 25., 0.]) * 2
    config.base_orientation_kd = np.array([5., 5., 5.]) * 2
    config.qp_foot_friction_coef = 0.7
    config.qp_weight_ddq = np.diag([1., 1., 10., 10., 10., 1.])
    config.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 1.5
    config.use_full_qp = False
    config.clip_grf_in_sim = True
    config.foot_friction = 0.7  # 0.7

    # Swing controller
    config.swing_foot_height = 0.12
    config.swing_foot_landing_clearance = 0.02

    # Termination condition
    config.terminate_on_body_contact = False
    config.terminate_on_limb_contact = False
    config.terminate_on_height = 0.15
    config.use_penetrating_contact = False

    # Reward
    config.rewards = [
        ('upright', 0.02),
        ('contact_consistency', 0.008),
        ('foot_slipping', 0.032),
        ('foot_clearance', 0.008),
        ('out_of_bound_action', 0.01),
        ('knee_contact', 0.064),
        ('stepping_freq', 0.008),
        ('com_distance_to_goal_squared', 0.016),
        ('com_height', 0.01),
    ]
    config.clip_negative_reward = False
    config.normalize_reward_by_phase = True

    config.terminal_rewards = []
    config.clip_negative_terminal_reward = False
    return config


def get_config():
    """Main entrance for the parsing the config"""
    config = ConfigDict()
    # config.training = ppo.get_training_config()     # Use PPO
    config.training = ddpg.get_training_config()  # Use DDPG
    config.env_class = go2_trot_env.Go2TrotEnv
    config.environment = get_env_config()
    return config
