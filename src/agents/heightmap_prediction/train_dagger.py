"""Trains student policies using DAgger."""
from absl import app
from absl import flags

from datetime import datetime
from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
from ml_collections.config_flags import config_flags
import numpy as np
import os
from rsl_rl.runners import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter

from src.agents.heightmap_prediction.replay_buffer import ReplayBuffer
from src.agents.heightmap_prediction.lstm_heightmap_predictor import LSTMHeightmapPredictor

from src.envs import env_wrappers

config_flags.DEFINE_config_file(
    "config", "src/agents/heightmap_prediction/configs/lstm_heightmap.py",
    "experiment configuration")
flags.DEFINE_string("logdir", "logs", "logdir")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")

FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused
  config = FLAGS.config

  # Setup logging
  logdir = os.path.join(FLAGS.logdir,
                        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  with open(os.path.join(logdir, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(config.to_yaml())

  writer = SummaryWriter(log_dir=logdir, flush_secs=10)

  # Initialize environment
  device = "cuda" if FLAGS.use_gpu else "cpu"
  env = config.env_class(num_envs=config.num_envs,
                         device=device,
                         config=config.env_config,
                         show_gui=FLAGS.show_gui)
  env = env_wrappers.RangeNormalize(env)

  # Initialize teacher policy
  runner = OnPolicyRunner(env,
                          config.teacher_config.training,
                          config.teacher_ckpt,
                          device=device)
  runner.load(config.teacher_ckpt)
  policy = runner.alg.actor_critic

  # Initialize heightmap predictor
  heightmap_predictor = LSTMHeightmapPredictor(
      dim_output=len(config.env_config.measured_points_x) *
      len(config.env_config.measured_points_y),
      vertical_res=config.env_config.camera_config.vertical_res,
      horizontal_res=config.env_config.camera_config.horizontal_res,
  ).to(device)

  # Initialize replay buffer
  replay_buffer = ReplayBuffer(env, device=device)
  rewards, cycles, _ = replay_buffer.collect_data(
      policy, heightmap_predictor=None, num_steps=config.num_init_steps)
  # replay_buffer.save(os.path.join(logdir, "replay_buffer.pt"))
  print(f"[Initial Rollout] Average Reward: {np.mean(rewards)}, "
        f"Average Cycle: {np.mean(cycles)}")
  # replay_buffer.save(os.path.join(logdir, "replay_buffer.pt"))
  for step in range(config.num_iters):
    # Train student policy
    loss = heightmap_predictor.train_on_data(replay_buffer,
                                             batch_size=config.batch_size,
                                             num_steps=config.num_steps)
    heightmap_predictor.save(os.path.join(logdir, f"model_{step}.pt"))
    # Collect more data
    rewards, cycles, _ = replay_buffer.collect_data(
        policy,
        heightmap_predictor=heightmap_predictor,
        num_steps=config.num_dagger_steps)
    # replay_buffer.save(os.path.join(logdir, "replay_buffer.pt"))

    # Log Rewards and Terrain Levels
    print(f"[Dagger step {step}] Average Reward: {np.mean(rewards)}, "
          f"Average Cycle: {np.mean(cycles)}")
    writer.add_scalar("Rollout/average_reward", np.mean(rewards), step)
    writer.add_scalar("Rollout/average_cycles", np.mean(cycles), step)
    writer.add_scalar("Heightmap Training/MSE", loss, step)


if __name__ == "__main__":
  app.run(main)
