"""Example of running the phase gait generator."""
from absl import app
from absl import flags

from typing import Sequence

import ml_collections
import numpy as np
import torch

from src.configs.defaults import sim_config
from src.controllers import phase_gait_generator
from src.robots import go1
from src.robots.motors import MotorCommand, MotorControlMode
from src.utils.torch_utils import to_torch

flags.DEFINE_integer("num_envs", 10, "number of environments to create.")
flags.DEFINE_float("total_time_secs", 2.,
                   "total amount of time to run the controller.")
FLAGS = flags.FLAGS


def create_sim(sim_conf):
  from isaacgym import gymapi, gymutil
  gym = gymapi.acquire_gym()
  _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
  if sim_conf.show_gui:
    graphics_device_id = sim_device_id
  else:
    graphics_device_id = -1

  sim = gym.create_sim(sim_device_id, graphics_device_id,
                       sim_conf.physics_engine, sim_conf.sim_params)

  if sim_conf.show_gui:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V,
                                        "toggle_viewer_sync")
  else:
    viewer = None

  plane_params = gymapi.PlaneParams()
  plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
  plane_params.static_friction = 1.
  plane_params.dynamic_friction = 1.
  plane_params.restitution = 0.
  gym.add_ground(sim, plane_params)
  return sim, viewer


def get_init_positions(num_envs: int, distance=1.) -> Sequence[float]:
  num_cols = int(np.sqrt(num_envs))
  init_positions = np.zeros((num_envs, 3))
  for idx in range(num_envs):
    init_positions[idx, 0] = idx // num_cols * distance
    init_positions[idx, 1] = idx % num_cols * distance
    init_positions[idx, 2] = 0.34
  return init_positions


def get_gait_config():
  config = ml_collections.ConfigDict()
  config.stepping_frequency = 1
  config.initial_offset = np.array([0.1, 0.1, 0., 0.]) * (2 * np.pi)
  config.swing_ratio = np.array([0.7, 0.7, 0.6, 0.6])
  return config


def main(argv):
  del argv  # unused
  sim_conf = sim_config.get_config(use_gpu=True,
                                   show_gui=False,
                                   use_real_robot=True)
  sim, viewer = create_sim(sim_conf)
  robot = go1.Go1(num_envs=FLAGS.num_envs,
                  init_positions=get_init_positions(FLAGS.num_envs),
                  sim=sim,
                  viewer=viewer,
                  sim_config=sim_conf,
                  motor_control_mode=MotorControlMode.HYBRID,
                  terrain=None)

  gait_config = get_gait_config()
  gait_generator = phase_gait_generator.PhaseGaitGenerator(robot, gait_config)

  robot.reset()
  num_envs, num_dof = robot.num_envs, robot.num_dof
  device = "cuda"
  dummy_command = MotorCommand(desired_position=torch.zeros(
      (num_envs, num_dof), device=device),
                               kp=torch.zeros((num_envs, num_dof),
                                              device=device),
                               desired_velocity=torch.zeros(
                                   (num_envs, num_dof), device=device),
                               kd=torch.zeros((num_envs, num_dof),
                                              device=device),
                               desired_extra_torque=torch.zeros(
                                   (num_envs, num_dof), device=device))

  while robot.time_since_reset[0] <= FLAGS.total_time_secs:
    robot.step(dummy_command)
    gait_generator.update()
    print("Time: {}".format(robot.time_since_reset))
    print("Desired Contact: {}".format(gait_generator.desired_contact_state))
    print("Progress: {}".format(gait_generator.normalized_phase))
    # input("Any Key...")


if __name__ == "__main__":
  app.run(main)
