"""Example of running the phase gait generator."""
from absl import app
from absl import flags

from datetime import datetime
import pickle
import time
from typing import Sequence

import ml_collections
import numpy as np
import os
import scipy
from tqdm import tqdm
from isaacgym import gymapi
import torch
torch.set_printoptions(precision=4, sci_mode=False)

from src.configs.defaults import sim_config
from src.controllers import phase_gait_generator
from src.controllers import qp_torque_optimizer
from src.controllers import raibert_swing_leg_controller
from src.robots import go1, go1_robot
from src.robots.motors import MotorCommand, MotorControlMode
from src.utils.torch_utils import to_torch

flags.DEFINE_integer("num_envs", 10, "number of environments to create.")
flags.DEFINE_float("total_time_secs", 2.,
                   "total amount of time to run the controller.")
flags.DEFINE_bool("use_gpu", True, "whether to show GUI.")
flags.DEFINE_bool("show_gui", True, "whether to show GUI.")
flags.DEFINE_bool("use_real_robot", False, "whether to run on real robot.")
flags.DEFINE_bool("save_traj", False, "whether to save trajectory.")
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
  plane_params.static_friction = 0.
  plane_params.dynamic_friction = 0.
  plane_params.restitution = 0.
  gym.add_ground(sim, plane_params)
  return sim, viewer


def get_init_positions(num_envs: int,
                       distance: float = 1.,
                       device: str = "cpu") -> Sequence[float]:
  num_cols = int(np.sqrt(num_envs))
  init_positions = np.zeros((num_envs, 3))
  for idx in range(num_envs):
    init_positions[idx, 0] = idx // num_cols * distance
    init_positions[idx, 1] = idx % num_cols * distance
    init_positions[idx, 2] = 0.27
  return to_torch(init_positions, device=device)


def get_gait_config():
  config = ml_collections.ConfigDict()
  config.stepping_frequency = 2  #1
  config.initial_offset = np.array([0., 0., 0., 0.],
                                   dtype=np.float32) * (2 * np.pi)
  config.swing_ratio = np.array([0., 0., 0., 0.], dtype=np.float32)
  return config


def main(argv):
  del argv  # unused
  sim_conf = sim_config.get_config(use_gpu=FLAGS.use_gpu,
                                   show_gui=FLAGS.show_gui,
                                   use_real_robot=FLAGS.use_real_robot)
  if not FLAGS.use_real_robot:
    sim, viewer = create_sim(sim_conf)
  else:
    sim, viewer = None, None

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
                      terrain=None)

  gait_config = get_gait_config()
  gait_generator = phase_gait_generator.PhaseGaitGenerator(robot, gait_config)
  swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot, gait_generator, foot_landing_clearance=0., foot_height=0.1)
  torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
      robot,
      gait_generator,
      desired_body_height=0.26,
      weight_ddq=np.diag([1., 1., 10., 10., 10., 1.]),
      body_mass=12.125,
      body_inertia=np.array([0.14, 0.35, 0.35]) * 1.5,
      foot_friction_coef=0.6,
      use_full_qp=True,
      base_position_kp=np.array([50., 50., 50]),
      base_position_kd=np.array([1., 1., 1.]),
      base_orientation_kp=np.array([50., 50., 0.]),
      base_orientation_kd=np.array([10., 10., 1.]),
  )

  # Ensure JIT compilation
  # for _ in range(3):
  #   torque_optimizer.get_action(
  #       gait_generator.desired_contact_state,
  #       swing_foot_position=swing_leg_controller.desired_foot_positions)

  robot.reset()
  swing_leg_controller.reset()
  torque_optimizer.reset()
  num_envs, num_dof = robot.num_envs, robot.num_dof
  steps_count = 0
  logs = []
  start_time = time.time()
  with torch.inference_mode():
    while robot.time_since_reset[0] <= FLAGS.total_time_secs:
      if FLAGS.use_real_robot:
        robot.state_estimator.update_foot_contact(  # pytype: disable=attribute-error
            gait_generator.desired_contact_state)  # pytype: disable=attribute-error
      gait_generator.update()
      swing_leg_controller.update()
      torque_optimizer.update()

      # print(lin_command, ang_command)
      desired_height = 0.26 + 0. * np.sin(
          2 * np.pi * robot.time_since_reset[0])
      desired_roll_rate = -np.deg2rad(0) * np.sin(
          2 * np.pi * robot.time_since_reset[0])
      desired_pitch_rate = -np.deg2rad(60) * np.sin(
          2 * np.pi * robot.time_since_reset[0])
      torque_optimizer.desired_base_position = [[0., 0., desired_height]]
      torque_optimizer.desired_linear_velocity = [[0., 0., 0.]]
      torque_optimizer.desired_base_orientation_rpy = [[0, 0, 0.]]
      torque_optimizer.desired_angular_velocity = [[
          desired_roll_rate, desired_pitch_rate, 0.
      ]]

      motor_action, desired_acc, solved_acc, _, _ = torque_optimizer.get_action(
          gait_generator.desired_contact_state,
          swing_foot_position=swing_leg_controller.desired_foot_positions)
      robot.step(motor_action)
      steps_count += 1

      if FLAGS.save_traj or FLAGS.use_real_robot:
        logs.append(
            dict(timestamp=robot.time_since_reset,
                 base_position=torch.clone(robot.base_position),
                 base_orientation_rpy=torch.clone(robot.base_orientation_rpy),
                 base_velocity=torch.clone(robot.base_velocity_body_frame),
                 base_angular_velocity=torch.clone(
                     robot.base_angular_velocity_body_frame),
                 motor_positions=torch.clone(robot.motor_positions),
                 motor_velocities=torch.clone(robot.motor_velocities),
                 motor_action=motor_action,
                 motor_torques=robot.motor_torques,
                 foot_contact_force=robot.foot_contact_forces,
                 desired_acc_body_frame=desired_acc,
                 solved_acc_body_frame=solved_acc,
                 foot_positions_in_base_frame=robot.
                 foot_positions_in_base_frame))
      if FLAGS.show_gui:
        robot.render()

    print("Wallclock time: {}".format(time.time() - start_time))
    if FLAGS.use_real_robot or FLAGS.save_traj:
      mode = "real" if FLAGS.use_real_robot else "sim"
      output_dir = (
          f"eval_{mode}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl")
      output_path = os.path.join("logs/20240301/pitch_test", output_dir)
      if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

      with open(output_path, "wb") as fh:
        pickle.dump(logs, fh)
      print(f"Data logged to: {output_path}")


if __name__ == "__main__":
  app.run(main)
