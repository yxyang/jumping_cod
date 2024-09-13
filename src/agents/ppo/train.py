"""Train PPO policy using implementation from RSL_RL."""
from absl import app
from absl import flags
# from absl import logging

from datetime import datetime
import os

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from ml_collections.config_flags import config_flags
from rsl_rl.runners import OnPolicyRunner

from src.envs import env_wrappers

config_flags.DEFINE_config_file(
    "config", "src/agents/ppo/configs/go1_single_jump_terrain.py",
    "experiment configuration.")
flags.DEFINE_integer("num_envs", 4096, "number of parallel environments.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
flags.DEFINE_string("logdir", "logs", "logdir.")
flags.DEFINE_string("load_checkpoint", None, "checkpoint to load.")
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  device = "cuda:0" if FLAGS.use_gpu else "cpu"
  config = FLAGS.config

  logdir = os.path.join(FLAGS.logdir, config.training.runner.experiment_name,
                        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  with open(os.path.join(logdir, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(config.to_yaml())

  env = config.env_class(num_envs=FLAGS.num_envs,
                         device=device,
                         config=config.environment,
                         show_gui=FLAGS.show_gui)
  env = env_wrappers.RangeNormalize(env)


  runner = OnPolicyRunner(env, config.training, logdir, device=device)
  if FLAGS.load_checkpoint:
    runner.load(FLAGS.load_checkpoint)
  env.reset()
  runner.learn(num_learning_iterations=config.training.runner.max_iterations,
               init_at_random_ep_len=True)


if __name__ == "__main__":
  app.run(main)
