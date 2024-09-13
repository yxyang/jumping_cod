"""Example of running the phase gait generator."""
from absl import app
from absl import flags

import time
from typing import Sequence

import ml_collections
import numpy as np
import scipy
from tqdm import tqdm
from isaacgym import gymapi
import torch

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
    init_positions[idx, 2] = 0.34
  return to_torch(init_positions, device=device)


def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0.6
  vy = 0.2
  wz = 0.8

  # time_points = (0, 1, 9, 10, 15, 20, 25, 30)
  # speed_points = ((0, 0, 0, 0), (0, 0.6, 0, 0), (0, 0.6, 0, 0), (vx, 0, 0, 0),
  #                 (0, 0, 0, -wz), (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

  time_points = (0, 5, 10, 15, 20, 25, 30)
  speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
                  (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="nearest",
                                     fill_value="extrapolate",
                                     axis=0)(t)

  return speed[0:3], speed[3]


def get_gait_config():
  config = ml_collections.ConfigDict()
  config.stepping_frequency = 2  #1
  config.initial_offset = np.array([0., 0.5, 0.5, 0.],
                                   dtype=np.float32) * (2 * np.pi)
  config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
  # config.initial_offset = np.array([0.15, 0.15, -0.35, -0.35]) * (2 * np.pi)
  # config.swing_ratio = np.array([0.6, 0.6, 0.6, 0.6])
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
      desired_body_height=0.32,
      weight_ddq=np.diag([1., 1., 1., 1., 1., 1.]),
      body_inertia=np.array([0.14, 0.35, 0.35]) * 1.5,
      foot_friction_coef=0.4,
      use_full_qp=False,
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

  start_time = time.time()
  pbar = tqdm(total=FLAGS.total_time_secs)
  with torch.inference_mode():
    while robot.time_since_reset[0] <= FLAGS.total_time_secs:
      if FLAGS.use_real_robot:
        robot.state_estimator.update_foot_contact( # pytype: disable=attribute-error
            gait_generator.desired_contact_state)  # pytype: disable=attribute-error
      gait_generator.update()
      swing_leg_controller.update()
      torque_optimizer.update()

      # Update speed comand
      lin_command, ang_command = _generate_example_linear_angular_speed(
          robot.time_since_reset[0].cpu())
      # print(lin_command, ang_command)
      torque_optimizer.desired_linear_velocity = [lin_command]
      torque_optimizer.desired_angular_velocity = [[0., 0., ang_command]]

      motor_action, _, _, _, _ = torque_optimizer.get_action(
          gait_generator.desired_contact_state,
          swing_foot_position=swing_leg_controller.desired_foot_positions)
      robot.step(motor_action)
      steps_count += 1
      pbar.update(0.002)

      if FLAGS.show_gui:
        robot.render()

    print("Wallclock time: {}".format(time.time() - start_time))


if __name__ == "__main__":
  app.run(main)
