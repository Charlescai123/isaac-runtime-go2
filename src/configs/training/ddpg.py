"""Configuration for DDPG Policy"""
from ml_collections import ConfigDict
from rsl_rl.modules.network import Network

from src.envs import go2_wild_env
import torch
import numpy as np


def get_training_config():
    """Config for training"""
    config = ConfigDict()
    config.seed = 1

    alg_config = ConfigDict()
    runner_config = ConfigDict()

    # Replay buffer
    alg_config.action_noise_scale = 0.1
    alg_config.init_noise_std = 0.1
    alg_config.storage_initial_size = 0
    alg_config.storage_size = 1e5
    alg_config.batch_count = 1
    alg_config.batch_size = 4096

    # Actor
    alg_config.actor_lr = 1e-4
    alg_config.actor_activations = ["elu", "elu", "elu", "tanh"]
    alg_config.actor_hidden_dims = [512, 256, 128]
    alg_config.actor_init_gain = 0.5
    alg_config.actor_input_normalization = True
    alg_config.actor_recurrent_layers = 1
    alg_config.actor_recurrent_module = Network.recurrent_module_lstm
    alg_config.actor_recurrent_tf_context_length = 64
    alg_config.actor_recurrent_tf_head_count = 8
    alg_config.actor_shared_dims = None
    alg_config._actor_input_size_delta = 0

    # Critic
    alg_config.critic_lr = 1e-3
    alg_config.critic_activations = ["elu", "elu", "elu", "linear"]
    alg_config.critic_hidden_dims = [512, 256, 128]
    alg_config.critic_init_gain = 0.5
    alg_config.critic_input_normalization = True
    alg_config.critic_recurrent_layers = 1
    alg_config.critic_recurrent_module = Network.recurrent_module_lstm
    alg_config.critic_recurrent_tf_context_length = 64
    alg_config.critic_recurrent_tf_head_count = 8
    alg_config.critic_shared_dims = None
    alg_config._critic_input_size_delta = 0

    # Others
    alg_config.polyak = 0.995
    alg_config.recurrent = False
    alg_config.return_steps = 1

    # Runner
    runner_config.policy_class_name = "ActorCritic"
    runner_config.algorithm_class_name = "DDPG"
    runner_config.num_steps_per_env = 100
    runner_config.save_interval = 50
    runner_config.experiment_name = "train_ddpg"
    runner_config.max_iterations = 5000

    # Integrate
    config.algorithm = alg_config
    config.runner = runner_config

    return config
