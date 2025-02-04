"""Train DDPG policy using implementation from RSL_RL."""

import time

from absl import app
from absl import flags
# from absl import logging
from isaacgym import gymapi, gymutil
from datetime import datetime
import os

from ml_collections.config_flags import config_flags
from rsl_rl.runners import OffPolicyRunner

from src.envs import env_wrappers

config_flags.DEFINE_config_file("config", "src/configs/trot.py", "experiment configuration.")
flags.DEFINE_integer("num_envs", 1, "number of parallel environments.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("enable_ha_teacher", True, "whether to enable the HA-Teacher.")
flags.DEFINE_bool("enable_pusher", False, "whether to enable the robot pusher.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_string("logdir", "logs", "logdir.")
flags.DEFINE_string("load_checkpoint", None, "checkpoint to load.")
FLAGS = flags.FLAGS


def main(argv):
    del argv  # unused
    device = "cuda" if FLAGS.use_gpu else "cpu"
    config = FLAGS.config

    logdir = os.path.join(FLAGS.logdir, config.training.runner.experiment_name,
                          datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(config.to_yaml())

    # HA-Teacher module
    if FLAGS.enable_ha_teacher:
        config.environment.ha_teacher.enable = True
        config.environment.ha_teacher.chi = 0.15
        config.environment.ha_teacher.tau = 50

    env = config.env_class(num_envs=FLAGS.num_envs,
                           device=device,
                           config=config.environment,
                           show_gui=FLAGS.show_gui,
                           use_real_robot=FLAGS.use_real_robot)

    # Robot pusher
    if FLAGS.enable_pusher:
        env._pusher.push_enable = True

    env = env_wrappers.RangeNormalize(env)

    ddpg_runner = OffPolicyRunner(env, config.training, logdir, device=device)
    if FLAGS.load_checkpoint:
        ddpg_runner.load(FLAGS.load_checkpoint)
    ddpg_runner.learn(num_learning_iterations=config.training.runner.max_iterations,
                      init_at_random_ep_len=True)


if __name__ == "__main__":
    app.run(main)
