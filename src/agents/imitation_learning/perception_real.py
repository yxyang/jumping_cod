"""Receives camera image and publishes embeddings."""
from absl import app
from absl import flags
from absl import logging

import os

from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import torch
import yaml

flags.DEFINE_string(
    "policy_ckpt",
    "logs/20240330/1_calibrated_camera/2024_03_30_00_52_29/"
    "model_23.pt", "policy_ckpt.")
FLAGS = flags.FLAGS


class DepthEmbeddingServer:
  """Receives camera image and publishes embeddings."""
  def __init__(self, policy_path, embedding_publisher, device="cpu"):
    self._embedding_publisher = embedding_publisher
    self._bridge = CvBridge()
    self._device = device

    # Load vision policy
    config_path = os.path.join(os.path.dirname(policy_path), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
      config = yaml.load(f, Loader=yaml.Loader)
    self._policy = config.student_policy_class(dim_state=27, dim_action=16)
    self._policy.load(policy_path)
    self._policy.eval()
    logging.info("Policy Loaded.")

  def compute_embedding(self, message):
    frame = self._bridge.imgmsg_to_cv2(message)
    frame = torch.from_numpy(frame)[None, None, ...].to(self._device)
    with torch.no_grad():
      embedding = self._policy.image_embedding(frame).flatten()

    embedding = embedding.cpu().numpy()
    embedding_msg = Float32MultiArray()
    embedding_msg.layout.dim = [MultiArrayDimension()]
    embedding_msg.layout.dim[0].size = embedding.size
    embedding_msg.layout.dim[0].stride = 1
    embedding_msg.layout.dim[0].label = "embedding"
    embedding_msg.data = embedding.tolist()

    # Publish the message
    self._embedding_publisher.publish(embedding_msg)
    # logging.info(f"Embedding computation time:{time.time() - start_time}")
    # rospy.loginfo(f"Embedding computation time:{time.time() - start_time}")


def main(argv):
  del argv  # unused

  rospy.init_node('embedding_server', anonymous=True)
  embedding_pub = rospy.Publisher("/camera/depth/embedding",
                                  Float32MultiArray,
                                  queue_size=1)

  embedding_server = DepthEmbeddingServer(FLAGS.policy_ckpt, embedding_pub)

  rospy.Subscriber("/camera/depth/cnn_input", Image,
                   embedding_server.compute_embedding)
  rospy.spin()


if __name__ == '__main__':
  app.run(main)
