"""Example of Go1 robots in Isaac Gym."""
# pytype: disable=attribute-error
from absl import app
from absl import flags

from datetime import datetime
import os
import pickle
from typing import Sequence

import numpy as np
from isaacgym import gymapi, gymutil  # Only import in sim
import torch
from tqdm import tqdm

from src.configs.defaults import sim_config, go1_camera_config
from src.robots import go1
from src.robots.motors import MotorCommand, MotorControlMode
from src.utils.torch_utils import to_torch

flags.DEFINE_string("logdir", None, "logdir")
FLAGS = flags.FLAGS


def create_sim(sim_conf):
  gym = gymapi.acquire_gym()
  _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
  graphics_device_id = sim_device_id
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


def get_init_positions(num_envs: int,
                       distance=1.,
                       device: str = "cpu") -> Sequence[float]:
  num_cols = int(np.sqrt(num_envs))
  init_positions = np.zeros((num_envs, 3))
  for idx in range(num_envs):
    init_positions[idx, 0] = idx // num_cols * distance
    init_positions[idx, 1] = idx % num_cols * distance
    init_positions[idx, 2] = 0.3
  return to_torch(init_positions, device=device)


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
  sim_conf = sim_config.get_config(use_gpu=False,
                                   show_gui=True,
                                   use_real_robot=False)
  sim, viewer = create_sim(sim_conf)

  camera_config = go1_camera_config.get_config()
  robot_class = go1.Go1
  robot = robot_class(num_envs=1,
                      init_positions=get_init_positions(
                          1, device=sim_conf.sim_device),
                      sim=sim,
                      viewer=viewer,
                      sim_config=sim_conf,
                      motor_control_mode=MotorControlMode.HYBRID,
                      terrain=None,
                      camera_config=camera_config)
  robot.reset()
  logs = []
  for _ in tqdm(range(2500)):
    action = get_action(robot,
                        robot.time_since_reset[0],
                        device=sim_conf.sim_device)
    robot.step(action)
    robot.step_graphics()
    robot.render()
    depth_image = robot.get_camera_image(0, mode="depth")
    depth_image = -torch.nan_to_num(depth_image, neginf=-3) / 3
    depth_image = depth_image.clip(min=0, max=1)
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
             foot_positions_in_base_frame=robot.foot_positions_in_base_frame,
             depth_image=depth_image))

  filename = (
      f"calibration_sim_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl")
  output_path = os.path.join(FLAGS.logdir, filename)
  if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)
  with open(output_path, "wb") as fh:
    pickle.dump(logs, fh)
  print(f"Data logged to: {output_path}")


if __name__ == "__main__":
  app.run(main)
