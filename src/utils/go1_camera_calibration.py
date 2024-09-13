"""Example of Go1 robots in Isaac Gym."""
# pytype: disable=attribute-error
from absl import app
from absl import flags

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

from src.configs.defaults import sim_config
from src.robots import go1_robot
from src.robots.motors import MotorCommand, MotorControlMode
from src.utils.torch_utils import to_torch

flags.DEFINE_string("logdir", None, "logdir")
FLAGS = flags.FLAGS


class DepthEmbeddingBuffer:
  """Buffer to store depth image embedding."""
  def __init__(self, dim_embedding: int = 120):
    self._dim_embedding = dim_embedding
    self._last_embedding = torch.zeros([1, dim_embedding])
    self._bridge = CvBridge()
    self._last_image = np.zeros((48, 60))
    self.has_new_image = False

  def update_image(self, msg):
    frame = self._bridge.imgmsg_to_cv2(msg)
    self._last_image = np.array(frame)
    self.has_new_image = True

  def update_embedding(self, msg):
    self._last_embedding = torch.from_numpy(
        np.array(msg.data, dtype=np.float32))[None, :]

  @property
  def last_embedding(self):
    return self._last_embedding

  @property
  def last_image(self):
    return self._last_image


def get_action(robot, t, device="cuda"):
  mid_action = to_torch([0.0, 0.9, -1.8] * 4, device=device)
  amplitude = to_torch([0.0, 0.2, -0.4] * 4, device=device)
  freq = 1.0
  num_envs, num_dof = robot.num_envs, robot.num_dof
  return MotorCommand(
      desired_position=torch.zeros((num_envs, num_dof), device=device) +
      mid_action + amplitude * torch.sin(2 * torch.pi * freq * t),
      kp=torch.zeros(
          (num_envs, num_dof), device=device) + robot.motor_group.kps,
      desired_velocity=torch.zeros((num_envs, num_dof), device=device),
      kd=torch.zeros(
          (num_envs, num_dof), device=device) + robot.motor_group.kds,
      desired_extra_torque=torch.zeros((num_envs, num_dof), device=device))


def main(argv):
  del argv  # unused

  rospy.init_node("eval_real")
  embedding_buffer = DepthEmbeddingBuffer()
  rospy.Subscriber("/camera/depth/embedding", Float32MultiArray,
                   embedding_buffer.update_embedding)
  rospy.Subscriber("/camera/depth/cnn_input", Image,
                   embedding_buffer.update_image)

  sim_conf = sim_config.get_config(use_gpu=False,
                                   show_gui=False,
                                   use_real_robot=True)

  robot = go1_robot.Go1Robot(num_envs=1,
                             init_positions=None,
                             sim=None,
                             viewer=None,
                             sim_config=sim_conf,
                             motor_control_mode=MotorControlMode.HYBRID,
                             terrain=None)
  robot.reset()
  logs = []
  start_time = time.time()
  while time.time() - start_time < 5:
    action = get_action(robot,
                        robot.time_since_reset[0],
                        device=sim_conf.sim_device)
    robot.step(action)

    logs.append(
        dict(timestamp=robot.time_since_reset,
             base_position=torch.clone(robot.base_position),
             base_orientation_rpy=torch.clone(robot.base_orientation_rpy),
             base_velocity=torch.clone(robot.base_velocity_body_frame),
             base_angular_velocity=torch.clone(
                 robot.base_angular_velocity_body_frame),
             motor_positions=torch.clone(robot.motor_positions),
             motor_velocities=torch.clone(robot.motor_velocities),
             motor_action=action,
             motor_torques=robot.motor_torques,
             foot_contact_force=robot.foot_contact_forces,
             foot_positions_in_base_frame=robot.foot_positions_in_base_frame))
    if embedding_buffer.has_new_image:
      embedding_buffer.has_new_image = False
      logs[-1]["depth_image"] = embedding_buffer.last_image.copy()

  print(f"Total time: {time.time() - start_time}")
  filename = (
      f"calibration_real_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl")
  output_path = os.path.join(FLAGS.logdir, filename)
  if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)
  with open(output_path, "wb") as fh:
    pickle.dump(logs, fh)
  print(f"Data logged to: {output_path}")

if __name__ == "__main__":
  app.run(main)
