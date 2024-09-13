"""Example of Go1 robots in Isaac Gym."""
# pytype: disable=attribute-error
# pytype: disable=protected-access
from absl import app
from absl import flags

import time

import numpy as np
import torch

from src.configs.defaults import sim_config
from src.robots import go1_robot
from src.robots.motors import MotorCommand, MotorControlMode

flags.DEFINE_string("logdir", None, "logdir")
FLAGS = flags.FLAGS


def test_robot(robot):
  # Sending zero torque commands to ensure robot connection
  for _ in range(10):
    robot.robot_interface.send_command(np.zeros(60, dtype=np.float32))
    time.sleep(0.001)

  kps = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                     device=robot.device)
  kds = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
                     device=robot.device)
  desired_torque = torch.tensor(
      [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -5.]],
      device=robot.device)

  print("About to test the robot.")
  # Stand up in 1.5 seconds, and fix the standing pose afterwards
  reset_time = 3
  start_time = time.time()
  while time.time() - start_time < reset_time:
    action = MotorCommand(desired_position=torch.zeros((robot.num_envs, 12),
                                                       device=robot.device),
                          kp=kps,
                          desired_velocity=torch.zeros((robot.num_envs, 12),
                                                       device=robot.device),
                          kd=kds,
                          desired_extra_torque=desired_torque)
    robot.step(action, MotorControlMode.HYBRID)


def main(argv):
  del argv  # unused
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
  test_robot(robot)


if __name__ == "__main__":
  app.run(main)
