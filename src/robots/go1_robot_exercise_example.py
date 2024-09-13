"""Example of Go1 robots in Isaac Gym."""
# pytype: disable=attribute-error
from absl import app
from absl import flags

import os
import time
from typing import Sequence

import numpy as np
from PIL import Image as im
import torch
from tqdm import tqdm

from src.configs.defaults import sim_config, go1_camera_config
from src.robots import go1, go1_robot
from src.robots.motors import MotorCommand, MotorControlMode
from src.utils.torch_utils import to_torch

flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("use_real_robot", False, "whether to run on real robot.")
flags.DEFINE_integer("num_envs", 10, "number of environments to create.")
flags.DEFINE_string("camera_image_dir", None, "where to store camera images.")
FLAGS = flags.FLAGS


def create_sim(sim_conf):
  from isaacgym import gymapi, gymutil # Only import in sim
  gym = gymapi.acquire_gym()
  _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
  if sim_conf.show_gui or FLAGS.camera_image_dir:
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


def get_init_positions(num_envs: int,
                       distance=1.,
                       device: str = "cpu") -> Sequence[float]:
  num_cols = int(np.sqrt(num_envs))
  init_positions = np.zeros((num_envs, 3))
  for idx in range(num_envs):
    init_positions[idx, 0] = idx // num_cols * distance
    init_positions[idx, 1] = idx % num_cols * distance
    init_positions[idx, 2] = 0.34
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
  sim_conf = sim_config.get_config(use_gpu=FLAGS.use_gpu,
                                   show_gui=FLAGS.show_gui,
                                   use_real_robot=FLAGS.use_real_robot)
  if not FLAGS.use_real_robot:
    sim, viewer = create_sim(sim_conf)
  else:
    sim, viewer = None, None

  camera_config = None
  if FLAGS.camera_image_dir:
    camera_config = go1_camera_config.get_config()
    if not os.path.exists(FLAGS.camera_image_dir):
      os.makedirs(FLAGS.camera_image_dir)

  if FLAGS.use_real_robot:
    robot_class = go1_robot.Go1Robot
  else:
    robot_class = go1.Go1

  robot = robot_class(num_envs=FLAGS.num_envs,
                      init_positions=get_init_positions(
                          FLAGS.num_envs, device=sim_conf.sim_device),
                      sim=sim,
                      viewer=viewer,
                      sim_config=sim_conf,
                      motor_control_mode=MotorControlMode.HYBRID,
                      terrain=None,
                      camera_config=camera_config)
  robot.reset()
  start_time = time.time()
  for step in tqdm(range(500)):
    action = get_action(robot,
                        robot.time_since_reset[0],
                        device=sim_conf.sim_device)
    robot.step(action)
    if FLAGS.use_real_robot:
      time.sleep(0.001)
    if FLAGS.show_gui:
      robot.render()

    if FLAGS.camera_image_dir:
      mode = "depth"
      for i in range(FLAGS.num_envs):
        image_arr = robot.get_camera_image(i, mode=mode)
        if mode == "RGBA":
          im.fromarray(image_arr, mode=mode).save(
              os.path.join(FLAGS.camera_image_dir, f"step_{step}_env{i}.png"))
        else:
          image_arr[image_arr == -np.inf] = 0
          image_arr[image_arr < -50] = -50
          normalized_depth = -255.0 * (image_arr / np.min(image_arr + 1e-4))
          normalized_depth_image = im.fromarray(normalized_depth.astype(
              np.uint8),
                                                mode="L")
          normalized_depth_image.save(
              os.path.join(FLAGS.camera_image_dir, f"step_{step}_env{i}.png"))

    # print("Foot: {}".format(robot.foot_positions_in_base_frame[0, :, 1]))
    # print("Base: {}".format(robot.base_angular_velocity_body_frame[0]))
  print(f"Total time: {time.time() - start_time}")


if __name__ == "__main__":
  app.run(main)
