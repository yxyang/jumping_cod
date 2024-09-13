"""Evaluate a trained policy."""
from absl import app
from absl import flags
# from absl import logging

import collections
from datetime import datetime
import os
import pickle
import time

import cv2
from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
import numpy as np
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

from src.agents.heightmap_prediction.lstm_heightmap_predictor import LSTMHeightmapPredictor
from src.envs import env_wrappers
from src.envs.terrain import GenerationMethod  # pylint: disable=unused-import
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string("logdir", None, "logdir.")
flags.DEFINE_bool("use_gpu", False, "whether to use GPU.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_integer("num_envs", 1,
                     "number of environments to evaluate in parallel.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")

FLAGS = flags.FLAGS


def display_depth(depth_array, depth_cutoff=1):
  # Normalize the depth array to 0-255 for visualization
  depth_array = -np.nan_to_num(depth_array.cpu(), neginf=-depth_cutoff).clip(
      -depth_cutoff, 0.)
  normalized_depth = ((depth_array / depth_cutoff) * 255).astype(np.uint8)
  # Apply colormap
  cv2.imshow("Depth Image", normalized_depth)
  cv2.waitKey(1)  # 1 millisecond delay

  # print(f"Depth min: {np.min(depth_array)}, max: {np.max(depth_array)}")


def main(argv):
  del argv  # unused

  device = "cuda" if FLAGS.use_gpu else "cpu"

  # Load config and policy
  config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
  policy_path = FLAGS.logdir
  root_path = os.path.dirname(FLAGS.logdir)

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    config.env_config.terrain.num_rows = 10
    config.env_config.terrain.num_cols = 1
    config.env_config.terrain.generation_method = \
        GenerationMethod.CURRICULUM
    config.env_config.terrain.curriculum = True
    config.env_config.terrain.min_init_level = 6
    config.env_config.terrain.max_init_level = 7
    config.env_config.terrain.terrain_proportions = dict(
        slope_smooth=0.,
        slope_rough=0.,
        stair=0.,
        obstacles=0.,
        stepping_stones=0.,
        gap=1.,
        pit=0.,
    )

  env = config.env_class(num_envs=FLAGS.num_envs,
                         device=device,
                         config=config.env_config,
                         show_gui=FLAGS.show_gui)
  env = env_wrappers.RangeNormalize(env)

  # env = env_wrappers.RangeNormalize(env)
  if FLAGS.use_real_robot:
    env.robot.state_estimator.use_external_contact_estimator = (
        not FLAGS.use_contact_sensor)

  # Initialize teacher policy
  runner = OnPolicyRunner(env,
                          config.teacher_config.training,
                          config.teacher_ckpt,
                          device=device)
  runner.load(config.teacher_ckpt)
  policy = runner.alg.actor_critic

  # Retrieve policy
  heightmap_predictor = LSTMHeightmapPredictor(
      dim_output=len(config.env_config.measured_points_x) *
      len(config.env_config.measured_points_y),
      vertical_res=config.env_config.camera_config.vertical_res,
      horizontal_res=config.env_config.camera_config.horizontal_res,
  ).to(device)
  heightmap_predictor.load(policy_path)
  heightmap_predictor.eval()

  env.reset()
  images = collections.deque(maxlen=5)
  for _ in range(20):
    # Reset environment
    # print(f"Level: {env.terrain_levels}")
    total_reward = torch.zeros(FLAGS.num_envs, device=device)
    steps_count = 0
    heightmap_predictor.reset()

    start_time = time.time()
    logs = []
    with torch.no_grad():
      while True:
        steps_count += 1
        env.robot.step_graphics()
        curr_imgs = []
        for env_id in range(FLAGS.num_envs):
          curr_imgs.append(
              to_torch(env.robot.get_camera_image(env_id, mode="depth"),
                       device=device))
        curr_imgs = torch.stack(curr_imgs, dim=0)
        images.append(curr_imgs)

        proprioceptive_state = env.get_proprioceptive_observation()
        height = heightmap_predictor.forward(
            base_state=env.get_perception_base_states(),
            foot_positions=(
                env.robot.foot_positions_in_gravity_frame[:, :, [0, 2]] *
                env.gait_generator.desired_contact_state_state_estimation[:, :,
                                                                          None]
            ).reshape((-1, 8)),
            depth_image=images[0])
        obs = torch.concatenate((proprioceptive_state, height), dim=-1)
        # obs[:, 10] = -height[:, 10]

        normalized_obs = env.normalize_observ(obs)
        action = policy.act_inference(normalized_obs)
        action = action.clip(min=env.action_space[0], max=env.action_space[1])

        display_depth(curr_imgs[0])
        # print(f"Base Height: {env.robot.base_position[0, 2]}")
        _, _, reward, done, info = env.step(
            action, base_height_override=-height[:, 10])
        total_reward += reward
        logs.extend(info["logs"])
        logs[-1]["depth_image"] = (-torch.nan_to_num(curr_imgs, neginf=-3) /
                                   3).clip(min=0, max=1)
        logs[-1]["height_est"] = torch.clone(height)

        if done.any():
          print(info["episode"])
          break
        input("Any Key...")

        if FLAGS.show_gui and steps_count % 4 == 0:
          env.robot.render()

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
