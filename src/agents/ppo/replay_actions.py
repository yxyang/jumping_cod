"""Evaluate a trained policy."""
from absl import app
from absl import flags
# from absl import logging

from datetime import datetime
import os
import pickle
import time

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
# from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

# from src.envs import env_wrappers
# from src.envs.terrain import GenerationMethod

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1,
                     "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused

  device = "cuda" if FLAGS.use_gpu else "cpu"

  # Load config and policy
  config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
  traj_path = FLAGS.logdir
  root_path = os.path.dirname(FLAGS.logdir)

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    # config.environment.qp_foot_friction_coef = 0.4
    # config.environment.terrain.type = "trimesh"
    # config.environment.terrain.terrain_length = 10
    # config.environment.terrain.terrain_width = 10
    # config.environment.terrain.border_size = 10
    # config.environment.terrain.num_rows = 1
    # config.environment.terrain.num_cols = 1
    # config.environment.terrain.horizontal_scale = 0.1
    # config.environment.terrain.vertical_scale = 0.005
    # config.environment.terrain.move_up_distance = 2
    # config.environment.terrain.slope_threshold = 0.75
    # config.environment.terrain.generation_method = GenerationMethod.CURRICULUM
    # config.environment.terrain.max_init_level = 1
    # config.environment.terrain.terrain_proportions = dict(
    #     slope_smooth=0.,
    #     slope_rough=1.,
    #     stair=0.,
    #     obstacles=0.,
    #     stepping_stones=0.,
    #     gap=0.,
    #     pit=0.,
    # )
    # desired_distance = .1
    # config.environment.goal_lb = torch.tensor([desired_distance, 0])
    # config.environment.goal_ub = torch.tensor([desired_distance, 0])
    config.environment.max_jumps = 6
    # config.environment.use_yaw_feedback = True

  with open(traj_path, "rb") as f:
    traj = pickle.load(f)

  env = config.env_class(num_envs=FLAGS.num_envs,
                         device=device,
                         config=config.environment,
                         show_gui=FLAGS.show_gui,
                         use_real_robot=FLAGS.use_real_robot)
  # env = env_wrappers.RangeNormalize(env)
  env.reset()
  total_reward = torch.zeros(FLAGS.num_envs, device=device)
  logs = []
  start_time = time.time()
  for frame in traj[::5]:
    action = frame["env_action"]
    # action = runner.alg.act(state, state)
    _, _, reward, done, info = env.step(action)
    # print(f"Ang Vel: {env.robot.base_angular_velocity_world_frame}")
    # print(f"Time: {env.robot.time_since_reset}, Reward: {reward}")
    # print(f"Action: {action}s")
    # print(f"Desired contact: {env.gait_generator.desired_contact_state}")
    # print(f"Foot contact: {env.robot.foot_contacts}")
    # print(f"Contact force: {env.robot.foot_contact_forces}")
    # input("Any Key...")

    total_reward += reward
    logs.extend(info["logs"])
    # print(f"Per-step time: {time.time() - start_time}")
    if done.any():
      print(info["episode"])
      break

  print(f"Total reward: {total_reward}")
  print(f"Time elapsed: {time.time() - start_time}")

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
