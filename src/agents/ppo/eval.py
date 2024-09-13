"""Evaluate a trained policy."""
from absl import app
from absl import flags
# from absl import logging

from datetime import datetime
import os
import pickle
import time

from isaacgym import gymapi  # pylint: disable=unused-import
import numpy as np  # pylint: disable=unused-import
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

from src.envs import env_wrappers
from src.envs.terrain import GenerationMethod  # pylint: disable=unused-import
from src.utils.torch_utils import to_torch  # pylint: disable=unused-import

torch.set_printoptions(precision=4, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1,
                     "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")

FLAGS = flags.FLAGS


def get_latest_policy_path(logdir):
  files = [
      entry for entry in os.listdir(logdir)
      if os.path.isfile(os.path.join(logdir, entry))
  ]
  files.sort(key=lambda entry: os.path.getmtime(os.path.join(logdir, entry)))
  files = files[::-1]

  for entry in files:
    if entry.startswith("model"):
      return os.path.join(logdir, entry)
  raise ValueError("No Valid Policy Found.")


def main(argv):
  del argv  # unused

  device = "cuda" if FLAGS.use_gpu else "cpu"

  # Load config and policy
  if FLAGS.logdir.endswith("pt"):
    config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
    policy_path = FLAGS.logdir
    root_path = os.path.dirname(FLAGS.logdir)
  else:
    # Find the latest policy ckpt
    config_path = os.path.join(FLAGS.logdir, "config.yaml")
    policy_path = get_latest_policy_path(FLAGS.logdir)
    root_path = FLAGS.logdir

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    # config.environment.terrain.type = "plane"
    if FLAGS.use_real_robot:
      config.environment.terrain.type = "plane"
      config.environment.terrain.curriculum = True
    # else:
    # config.environment.terrain.type = "trimesh"
    # config.environment.terrain.border_size = 10
    config.environment.terrain.num_rows = 10
    config.environment.terrain.num_cols = 1
    # config.environment.terrain.horizontal_scale = 0.1
    # config.environment.terrain.vertical_scale = 0.005
    # config.environment.terrain.move_up_distance = 2
    # config.environment.terrain.slope_threshold = 0.75
    # config.environment.terrain.generation_method = \
    #     GenerationMethod.UNIFORM_RANDOM
    config.environment.terrain.min_init_level = 6
    config.environment.terrain.max_init_level = 7

    config.environment.terrain.terrain_proportions = dict(
        slope_smooth=0.,
        slope_rough=0.,
        stair=1.,
        obstacles=0.,
        stepping_stones=0.,
        gap=0.,
        pit=0.,
        inv_pit=0.,
        hurdle=0.
    )

  env = config.env_class(num_envs=FLAGS.num_envs,
                         device=device,
                         config=config.environment,
                         show_gui=FLAGS.show_gui,
                         use_real_robot=FLAGS.use_real_robot)
  env = env_wrappers.RangeNormalize(env)
  if FLAGS.use_real_robot:
    env.robot.state_estimator.use_external_contact_estimator = (
        not FLAGS.use_contact_sensor)

  # Retrieve policy
  runner = OnPolicyRunner(env, config.training, policy_path, device=device)
  runner.load(policy_path)
  policy = runner.alg.actor_critic
  runner.alg.actor_critic.train()

  state, _ = env.reset()
  for _ in range(5):
    # Reset environment
    # print(f"Level: {env.terrain_levels}")
    total_reward = torch.zeros(FLAGS.num_envs, device=device)
    steps_count = 0
    policy.reset()

    start_time = time.time()
    logs = []
    with torch.no_grad():
      while True:
        steps_count += 1
        action = policy.act_inference(state)
        state, _, reward, done, info = env.step(action)

        total_reward += reward
        logs.extend(info["logs"])
        # print(f"Per-step time: {time.time() - start_time}")
        if done.any():
          print(info["episode"])
          break

        if FLAGS.show_gui and steps_count % 4 == 0:
          env.robot.render()

    print(f"Total reward: {total_reward}")
    print(f"Time elapsed: {time.time() - start_time}")
    print(f"Total steps: {steps_count}")
    if FLAGS.use_real_robot or FLAGS.save_traj:
      mode = "real" if FLAGS.use_real_robot else "sim"
      output_dir = (
          f"eval_{mode}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl")
      output_path = os.path.join(root_path, output_dir)

      with open(output_path, "wb") as fh:
        pickle.dump(logs, fh)
      print(f"Data logged to: {output_path}")


if __name__ == "__main__":
  app.run(main)
