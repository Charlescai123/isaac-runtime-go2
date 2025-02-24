"""Train PPO policy using implementation from RSL_RL."""
import logging

import numpy as np
from absl import app
from absl import flags
# from absl import logging

from datetime import datetime
import os

from isaacgym import gymapi
from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
import torch

from ml_collections.config_flags import config_flags
from rsl_rl.runners import OnPolicyRunner

from src.configs.training import ppo
from src.envs import env_wrappers

config_flags.DEFINE_config_file(
    "config", "src/configs/wild_env_config.py", "experiment configuration.")
flags.DEFINE_integer("num_envs", 1, "number of parallel environments.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("enable_ha_teacher", False, "whether to enable the HA-Teacher.")
flags.DEFINE_bool("enable_pusher", False, "whether to enable the robot pusher.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_string("logdir", "logs", "logdir.")
flags.DEFINE_string("load_checkpoint", None, "checkpoint to load.")
flags.DEFINE_string("experiment_name", None, "experimental name to set.")
FLAGS = flags.FLAGS


def main(argv):
    del argv  # unused
    logging.disable(logging.CRITICAL)  # logging output

    device = "cuda" if FLAGS.use_gpu else "cpu"
    config = FLAGS.config

    # Experimental name set
    if FLAGS.experiment_name:
        config.training.runner.experiment_name = FLAGS.experiment_name

    logdir = os.path.join(FLAGS.logdir, config.training.runner.experiment_name,
                          datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(config.to_yaml())

    # Use PPO
    config.training = ppo.get_training_config()

    # HA-Teacher module
    if FLAGS.enable_ha_teacher:
        config.environment.ha_teacher.enable = True
        # config.environment.ha_teacher.chi = 0.1
        # config.environment.ha_teacher.tau = 100

    env = config.env_class(num_envs=FLAGS.num_envs,
                           device=device,
                           config=config.environment,
                           show_gui=FLAGS.show_gui,
                           use_real_robot=FLAGS.use_real_robot)
    # Robot pusher
    if FLAGS.enable_pusher:
        env._pusher.push_enable = True

    print(f"env_class: {config.env_class}")
    env = env_wrappers.RangeNormalize(env)

    mean_pos = torch.min(env.robot.base_position_world, dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
    target_pos = torch.mean(env.robot.base_position_world, dim=0).cpu().numpy() + np.array([0., 2., -0.5])
    cam_pos = gymapi.Vec3(*mean_pos)
    cam_target = gymapi.Vec3(*target_pos)
    env.robot._gym.viewer_camera_look_at(env.robot._viewer, None, cam_pos, cam_target)
    env.robot._gym.step_graphics(env.robot._sim)
    env.robot._gym.draw_viewer(env.robot._viewer, env.robot._sim, True)

    runner = OnPolicyRunner(env, config.training, logdir, device=device)
    if FLAGS.load_checkpoint:
        runner.load(FLAGS.load_checkpoint)
    runner.learn(num_learning_iterations=config.training.runner.max_iterations,
                 init_at_random_ep_len=False)


if __name__ == "__main__":
    app.run(main)
