"""Evaluate a trained policy."""
from absl import app
from absl import flags
# from absl import logging

from datetime import datetime
import os
import pickle
from sensor_msgs.msg import Image
import time

from cv_bridge import CvBridge
import numpy as np
import rospy
from rsl_rl.runners import OnPolicyRunner
import torch
import yaml

from src.agents.heightmap_prediction.lstm_heightmap_predictor import LSTMHeightmapPredictor
from src.envs import env_wrappers
from src.envs.terrain import GenerationMethod  # pylint: disable=unused-import
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string(
    "policy_ckpt", "logs/20240531/3_distill_5/"
    "2024_05_31_17_01_51/model_29.pt", "policy_ckpt.")
flags.DEFINE_bool("use_contact_sensor", False,
                  "whether to use contact sensor.")

FLAGS = flags.FLAGS


class DepthImageBuffer:
  """Buffer to store depth image embedding."""
  def __init__(self):
    self._bridge = CvBridge()
    self._last_image = torch.zeros((1, 48, 60))
    self._last_image_time = time.time()

  def update_image(self, msg):
    frame = np.array(self._bridge.imgmsg_to_cv2(msg))
    self._last_image = torch.from_numpy(frame)[None, ...]
    self._last_image_time = time.time()

  @property
  def last_image(self):
    return self._last_image

  @property
  def last_image_time(self):
    return self._last_image_time


def main(argv):
  del argv  # unused

  device = "cpu"
  # Load config and policy
  config_path = os.path.join(os.path.dirname(FLAGS.policy_ckpt), "config.yaml")
  policy_path = FLAGS.policy_ckpt
  root_path = os.path.dirname(FLAGS.policy_ckpt)
  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    config.env_config.terrain.type = "plane"
    config.env_config.terrain.curriculum = False
    config.env_config.max_jumps = 10
    config.env_config.use_full_qp = False

  # Initialize depth image topic subscriber
  rospy.init_node("eval_real")
  image_buffer = DepthImageBuffer()
  rospy.Subscriber("/camera/depth/cnn_input",
                   Image,
                   image_buffer.update_image,
                   queue_size=1,
                   tcp_nodelay=True)

  env = config.env_class(num_envs=1,
                         device=device,
                         config=config.env_config,
                         show_gui=False,
                         use_real_robot=True)
  env = env_wrappers.RangeNormalize(env)
  env.robot.state_estimator.use_external_contact_estimator = (
      not FLAGS.use_contact_sensor)

  # Retrieve policy
  runner = OnPolicyRunner(env,
                          config.teacher_config.training,
                          config.teacher_ckpt,
                          device=device)
  runner.load(config.teacher_ckpt)
  policy = runner.alg.actor_critic
  policy.eval()

  # Heightmap Predictor
  heightmap_predictor = LSTMHeightmapPredictor(
      dim_output=len(config.env_config.measured_points_x) *
      len(config.env_config.measured_points_y),
      vertical_res=config.env_config.camera_config.vertical_res,
      horizontal_res=config.env_config.camera_config.horizontal_res,
  ).to(device)
  heightmap_predictor.load(policy_path)
  heightmap_predictor.eval()

  env.reset()
  # Reset environment
  # print(f"Level: {env.terrain_levels}")
  total_reward = torch.zeros(1, device=device)
  steps_count = 0
  policy.reset()

  start_time = time.time()
  logs = []
  with torch.no_grad():
    while True:
      steps_count += 1
      proprioceptive_state = env.get_proprioceptive_observation()
      height = heightmap_predictor.forward_normalized(
          base_state=env.get_perception_base_states(),
          foot_positions=(
              env.robot.foot_positions_in_gravity_frame[:, :, [0, 2]] *
              env.gait_generator.desired_contact_state_state_estimation[:, :,
                                                                        None]
          ).reshape((-1, 8)),
          depth_image_normalized=image_buffer.last_image)
      obs = torch.concatenate((proprioceptive_state, height), dim=-1)
      # obs[:, 10] = -height[:, 10]
      normalized_obs = env.normalize_observ(obs)
      action = policy.act_inference(normalized_obs)
      action = action.clip(min=env.action_space[0], max=env.action_space[1])

      _, _, reward, done, info = env.step(action,
                                          base_height_override=-height[:, 10])
      total_reward += reward
      info["logs"][-1]["depth_image"] = image_buffer.last_image
      info["logs"][-1]["depth_image_ts"] = image_buffer.last_image_time
      info["logs"][-1]["height_est"] = torch.clone(height)
      logs.extend(info["logs"])

      if done.any():
        print(info["episode"])
        break

  print(f"Total reward: {total_reward}")
  print(f"Time elapsed: {time.time() - start_time}")
  output_dir = (
      f"eval_real_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl")
  output_path = os.path.join(root_path, output_dir)

  with open(output_path, "wb") as fh:
    pickle.dump(logs, fh)
  print(f"Data logged to: {output_path}")


if __name__ == "__main__":
  app.run(main)
