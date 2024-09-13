"""Evaluate a trained policy."""
from absl import app
from absl import flags
# from absl import logging

from datetime import datetime
import os
import pickle
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import time

from cv_bridge import CvBridge
import numpy as np
import rospy
import torch
import yaml

from src.envs import env_wrappers
from src.envs.terrain import GenerationMethod  # pylint: disable=unused-import
torch.set_printoptions(precision=2, sci_mode=False)

flags.DEFINE_string(
    "logdir", "logs/20240330/1_calibrated_camera/2024_03_30_00_52_29/"
    "model_23.pt", "policy_ckpt.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
flags.DEFINE_bool("use_contact_sensor", True, "whether to use contact sensor.")

FLAGS = flags.FLAGS


class DepthEmbeddingBuffer:
  """Buffer to store depth image embedding."""
  def __init__(self, dim_embedding: int = 120):
    self._dim_embedding = dim_embedding
    self._last_embedding = torch.zeros([1, dim_embedding])
    self._bridge = CvBridge()
    self._last_image = np.zeros((48, 60))
    self._last_image_time = time.time()
    self._last_embedding_time = time.time()

  def update_image(self, msg):
    frame = self._bridge.imgmsg_to_cv2(msg)
    self._last_image = np.array(frame)
    self._last_image_time = time.time()

  def update_embedding(self, msg):
    self._last_embedding = torch.from_numpy(
        np.array(msg.data, dtype=np.float32))[None, :]
    self._last_embedding_time = time.time()

  @property
  def last_embedding(self):
    return self._last_embedding

  @property
  def last_embedding_time(self):
    return self._last_embedding_time

  @property
  def last_image(self):
    return self._last_image

  @property
  def last_image_time(self):
    return self._last_image_time


def main(argv):
  del argv  # unused

  rospy.init_node("eval_real")
  embedding_buffer = DepthEmbeddingBuffer()
  rospy.Subscriber("/camera/depth/embedding",
                   Float32MultiArray,
                   embedding_buffer.update_embedding,
                   queue_size=1,
                   tcp_nodelay=True)
  rospy.Subscriber("/camera/depth/cnn_input",
                   Image,
                   embedding_buffer.update_image,
                   queue_size=1,
                   tcp_nodelay=True)

  device = "cpu"

  # Load config and policy
  config_path = os.path.join(os.path.dirname(FLAGS.logdir), "config.yaml")
  policy_path = FLAGS.logdir
  root_path = os.path.dirname(FLAGS.logdir)

  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.Loader)

  with config.unlocked():
    config.env_config.terrain.type = "plane"
    config.env_config.terrain.curriculum = False
    config.env_config.max_jumps = 8

    # config.env_config.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 6

  env = config.env_class(num_envs=1,
                         device=device,
                         config=config.env_config,
                         show_gui=False,
                         use_real_robot=True)
  env = env_wrappers.RangeNormalize(env)
  env.robot.state_estimator.use_external_contact_estimator = (
      not FLAGS.use_contact_sensor)

  # Retrieve policy
  policy = config.student_policy_class(
      dim_state=env.observation_space[0].shape[0] -
      len(config.env_config.measured_points_x) *
      len(config.env_config.measured_points_y),
      dim_action=env.action_space[0].shape[0]).to(device)
  policy.load(policy_path)
  policy.eval()

  state, _ = env.reset()
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
      action = policy.forward_with_embedding(embedding_buffer.last_embedding,
                                             state[:, :-440]).clip(
                                                 min=env.action_space[0],
                                                 max=env.action_space[1])

      state, _, reward, done, info = env.step(action)
      total_reward += reward
      info["logs"][-1]["depth_image"] = embedding_buffer.last_image
      info["logs"][-1]["depth_image_ts"] = embedding_buffer.last_image_time
      info["logs"][-1]["depth_embedding"] = embedding_buffer.last_embedding
      info["logs"][-1][
          "depth_embedding_ts"] = embedding_buffer.last_embedding_time
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
