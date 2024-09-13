"""Real robot class for Go1 robot."""
import time
from typing import Any, Optional, Union

import ml_collections
import numpy as np
import pybullet as p
import torch

import go1_interface
from src.robots import robot_state_estimator
from src.robots.motors import MotorCommand, MotorControlMode, MotorGroup, MotorModel
from src.utils.torch_utils import to_torch


class Go1Robot:
  """Go1 robot class."""
  def __init__(self,
               sim: Any,
               viewer: Any,
               num_envs: int,
               init_positions: torch.Tensor,
               sim_config: ml_collections.ConfigDict,
               motor_control_mode: MotorControlMode,
               terrain: Any,
               motor_torque_delay_steps: int = 0,
               camera_config: Union[ml_collections.ConfigDict, None] = None,
               randomize_com: bool = False):
    del motor_torque_delay_steps  # unused
    del sim  # unused
    del init_positions  # unused
    del terrain  # unused
    del randomize_com  # unused
    if num_envs != 1:
      raise ValueError("Only 1 real robot is supported at this time.")
    if viewer is not None:
      raise ValueError("Cannot enable GUI for real robot.")

    self._sim_config = sim_config
    self._num_envs = num_envs
    self._device = self._sim_config.sim_device
    self._num_dof = 12

    self._raw_state = go1_interface.LowState()
    self._contact_force_threshold = np.zeros(4)
    self._robot_interface = go1_interface.RobotInterface(0xff)
    self._state_estimator = robot_state_estimator.RobotStateEstimator(
        self, use_external_contact_estimator=False)
    self._last_reset_time = time.time()
    self._motor_control_mode = motor_control_mode
    self._foot_contact_force_histories = [[], [], [], []]

    self._motors = MotorGroup(device=sim_config.sim_device,
                              num_envs=num_envs,
                              motors=(
                                  MotorModel(
                                      name="FR_hip_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=0.0,
                                      min_position=-0.863,
                                      max_position=0.863,
                                      min_velocity=-30,
                                      max_velocity=30,
                                      min_torque=-23.7,
                                      max_torque=23.7,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=4.22,
                                  ),
                                  MotorModel(
                                      name="FR_thigh_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=0.9,
                                      min_position=-0.686,
                                      max_position=4.501,
                                      min_velocity=-30,
                                      max_velocity=30,
                                      min_torque=-23.7,
                                      max_torque=23.7,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=4.22,
                                  ),
                                  MotorModel(
                                      name="FR_calf_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=-1.8,
                                      min_position=-2.818,
                                      max_position=-0.888,
                                      min_velocity=-20,
                                      max_velocity=20,
                                      min_torque=-35.55,
                                      max_torque=35.55,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=6.33,
                                  ),
                                  MotorModel(
                                      name="FL_hip_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=0.0,
                                      min_position=-0.863,
                                      max_position=0.863,
                                      min_velocity=-30,
                                      max_velocity=30,
                                      min_torque=-23.7,
                                      max_torque=23.7,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=4.22,
                                  ),
                                  MotorModel(
                                      name="FL_thigh_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=0.9,
                                      min_position=-0.686,
                                      max_position=4.501,
                                      min_velocity=-30,
                                      max_velocity=30,
                                      min_torque=-23.7,
                                      max_torque=23.7,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=4.22,
                                  ),
                                  MotorModel(
                                      name="FL_calf_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=-1.8,
                                      min_position=-2.818,
                                      max_position=-0.888,
                                      min_velocity=-20,
                                      max_velocity=20,
                                      min_torque=-35.55,
                                      max_torque=35.55,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=6.33,
                                  ),
                                  MotorModel(
                                      name="RR_hip_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=0.0,
                                      min_position=-0.863,
                                      max_position=0.863,
                                      min_velocity=-30,
                                      max_velocity=30,
                                      min_torque=-23.7,
                                      max_torque=23.7,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=4.22,
                                  ),
                                  MotorModel(
                                      name="RR_thigh_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=0.9,
                                      min_position=-0.686,
                                      max_position=4.501,
                                      min_velocity=-30,
                                      max_velocity=30,
                                      min_torque=-23.7,
                                      max_torque=23.7,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=4.22,
                                  ),
                                  MotorModel(
                                      name="RR_calf_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=-1.8,
                                      min_position=-2.818,
                                      max_position=-0.888,
                                      min_velocity=-20,
                                      max_velocity=20,
                                      min_torque=-35.55,
                                      max_torque=35.55,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=6.33,
                                  ),
                                  MotorModel(
                                      name="RL_hip_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=0.0,
                                      min_position=-0.863,
                                      max_position=0.863,
                                      min_velocity=-30,
                                      max_velocity=30,
                                      min_torque=-23.7,
                                      max_torque=23.7,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=4.22,
                                  ),
                                  MotorModel(
                                      name="RL_thigh_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=0.9,
                                      min_position=-0.686,
                                      max_position=4.501,
                                      min_velocity=-30,
                                      max_velocity=30,
                                      min_torque=-23.7,
                                      max_torque=23.7,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=4.22,
                                  ),
                                  MotorModel(
                                      name="RL_calf_joint",
                                      motor_control_mode=motor_control_mode,
                                      init_position=-1.8,
                                      min_position=-2.818,
                                      max_position=-0.888,
                                      min_velocity=-20,
                                      max_velocity=20,
                                      min_torque=-35.55,
                                      max_torque=35.55,
                                      kp=100,
                                      kd=1,
                                      reduction_ratio=6.33,
                                  ),
                              ),
                              torque_delay_steps=0)

    self._hip_offset = to_torch(
        [[0.1881, -0.04675, 0.], [0.1881, 0.04675, 0.],
         [-0.1881, -0.04675, 0.], [-0.1881, 0.04675, 0.]],
        device=sim_config.sim_device)

    hip_position_single = to_torch((
        (0.1881, -0.13, 0),
        (0.1881, 0.13, 0),
        (-0.1881, -0.13, 0),
        (-0.1881, 0.13, 0),
    ),
                                   device=sim_config.sim_device)
    self._hip_positions_in_body_frame = torch.stack([hip_position_single] *
                                                    num_envs,
                                                    dim=0)
    self._post_physics_step()

  def update_desired_foot_contact(self, desired_contact):
    foot_forces = self.foot_contact_forces_numpy
    desired_contact = desired_contact.cpu().numpy().flatten()
    for idx in range(4):
      if desired_contact[idx]:
        if len(self._foot_contact_force_histories[idx]) > 0:
          self._contact_force_threshold[idx] = np.mean(
              self._foot_contact_force_histories[idx]) + 0.5 * np.std(
                  self._foot_contact_force_histories[idx])
          self._foot_contact_force_histories[idx] = []
      else:
        self._foot_contact_force_histories[idx].append(foot_forces[idx])

  def reset(self, reset_time: float = 1.5):
    # Sending zero torque commands to ensure robot connection
    for _ in range(10):
      self._robot_interface.send_command(np.zeros(60, dtype=np.float32))
      time.sleep(0.001)
      self._receive_observation()

    print("About to reset the robot.")
    initial_motor_position = self.motor_positions_numpy
    end_motor_position = self._motors.init_positions
    # Stand up in 1.5 seconds, and fix the standing pose afterwards
    standup_time = min(reset_time, 1.)
    stand_foot_forces = []
    for t in np.arange(0, reset_time,
                       self._sim_config.dt * self._sim_config.action_repeat):
      blend_ratio = min(t / standup_time, 1)
      desired_motor_position = blend_ratio * end_motor_position + (
          1 - blend_ratio) * initial_motor_position
      action = MotorCommand(desired_position=to_torch(
          desired_motor_position[None, :], device=self._device),
                            kp=self._motors.kps,
                            desired_velocity=torch.zeros((self._num_envs, 12),
                                                         device=self._device),
                            kd=self._motors.kds)
      self.step(action, MotorControlMode.POSITION)
      time.sleep(self._sim_config.dt * self._sim_config.action_repeat)
      if t > standup_time:
        stand_foot_forces.append(self.foot_contact_forces_numpy)

      # Calibrate foot force sensors
    stand_foot_forces = np.mean(stand_foot_forces, axis=0)
    self._contact_force_threshold = stand_foot_forces * 0.8

    self._last_reset_time = time.time()
    self._state_estimator.reset()
    self._post_physics_step()

  def reset_idx(self, env_ids):
    del env_ids  # unused
    self.reset()

  def _apply_action(
      self,
      action: MotorCommand,
      motor_control_mode: Optional[MotorControlMode] = None) -> None:
    if motor_control_mode is None:
      motor_control_mode = self._motor_control_mode
    command = np.zeros(60, dtype=np.float32)
    if motor_control_mode == MotorControlMode.POSITION:
      for motor_id in range(self._num_dof):
        command[motor_id * 5] = action.desired_position.cpu().numpy()[0,
                                                                      motor_id]
        command[motor_id * 5 + 1] = action.kp.cpu().numpy()[motor_id]
        command[motor_id * 5 + 3] = action.kd.cpu().numpy()[motor_id]
    elif motor_control_mode == MotorControlMode.HYBRID:
      command[0::5] = action.desired_position[0].cpu().numpy()
      command[1::5] = action.kp[0].cpu().numpy()
      command[2::5] = action.desired_velocity[0].cpu().numpy()
      command[3::5] = action.kd[0].cpu().numpy()
      command[4::5] = action.desired_extra_torque[0].cpu().numpy().clip(
          -35, 35)
    else:
      raise ValueError(
          f"Unknown motor control mode for Go1 robot: {motor_control_mode}.")

    self._robot_interface.send_command(command)

  def _receive_observation(self) -> None:
    self._raw_state = self._robot_interface.receive_observation()

  def step(self,
           action: MotorCommand,
           motor_control_mode: Optional[MotorControlMode] = None,
           perturbation_forces=None):
    del perturbation_forces  # unused
    for _ in range(self._sim_config.action_repeat):
      self._apply_action(action, motor_control_mode)
      self._receive_observation()
      self._post_physics_step()
      self._state_estimator.update(self._raw_state)

  def _post_physics_step(self):
    q = self._raw_state.imu.quaternion
    base_quat = np.array([q[1], q[2], q[3], q[0]])
    if np.sum(np.square(base_quat)) == 0:
      base_quat[-1] = 1
    self._base_rot_mat = np.array(
        p.getMatrixFromQuaternion(base_quat)).reshape((3, 3))
    self._base_rot_mat_zero_yaw = np.array(
        p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler(
                [self._raw_state.imu.rpy[0], self._raw_state.imu.rpy[1],
                 0]))).reshape((3, 3))
    self._base_rot_mat_torch = to_torch(self._base_rot_mat[None, :, :],
                                        device=self._device)
    self._base_rot_mat_t_torch = to_torch((self._base_rot_mat.T)[None, :, :],
                                          device=self._device)
    self._base_rot_mat_zero_yaw_torch = to_torch(
        self._base_rot_mat_zero_yaw[None, :, :], device=self._device)
    self._base_rot_mat_zero_yaw_t_torch = to_torch(
        (self._base_rot_mat_zero_yaw.T)[None, :, :], device=self._device)
    self._base_quat_torch = to_torch([base_quat], device=self._device)

    self._jacobians = self._compute_all_foot_jacobian()

    self._foot_positions_in_base_frame = (self._foot_positions_in_hip_frame() +
                                          self._hip_offset.cpu().numpy())
    self._foot_positions_in_gravity_frame = self._base_rot_mat_zero_yaw.dot(
        self._foot_positions_in_base_frame.T).T

  def compute_foot_jacobian(self, leg_id):
    return self._jacobians[leg_id * 3:leg_id * 3 + 3,
                           leg_id * 3:leg_id * 3 + 3]

  def _foot_positions_in_hip_frame(self):
    motor_positions = self.motor_positions_numpy.reshape((4, 3))
    theta_ab = motor_positions[:, 0]
    theta_hip = motor_positions[:, 1]
    theta_knee = motor_positions[:, 2]
    l_up = 0.213
    l_low = 0.213
    l_hip = np.array([-1, 1, -1, 1]) * 0.08
    leg_distance = np.sqrt(l_up**2 + l_low**2 +
                           2 * l_up * l_low * np.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * np.sin(eff_swing)
    off_z_hip = -leg_distance * np.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = np.cos(theta_ab) * off_y_hip - np.sin(theta_ab) * off_z_hip
    off_z = np.sin(theta_ab) * off_y_hip + np.cos(theta_ab) * off_z_hip
    return np.stack([off_x, off_y, off_z], axis=1)

  def get_motor_angles_from_foot_positions(self, foot_local_positions):
    foot_positions_in_hip_frame = foot_local_positions - self.hip_offset
    foot_positions_in_hip_frame = foot_positions_in_hip_frame.cpu().numpy()[0]

    l_up = 0.213
    l_low = 0.213
    l_hip = np.array([-1, 1, -1, 1]) * 0.08
    x = foot_positions_in_hip_frame[:, 0]
    y = foot_positions_in_hip_frame[:, 1]
    z = foot_positions_in_hip_frame[:, 2]
    theta_knee = -np.arccos(
        np.clip((x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
                (2 * l_low * l_up), -1, 1))
    l = np.sqrt(
        np.clip(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee),
                1e-7, 1))
    theta_hip = np.arcsin(np.clip(-x / l, -1, 1)) - theta_knee / 2
    c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
    s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
    theta_ab = np.arctan2(s1, c1)
    joint_angles = np.stack([theta_ab, theta_hip, theta_knee],
                            axis=1).flatten()
    return to_torch(joint_angles[None, :], device=self._device)

  def _compute_all_foot_jacobian(self):
    motor_positions = self.motor_positions_numpy.reshape((4, 3))
    l_up = 0.213
    l_low = 0.213
    l_hip = np.array([-1, 1, -1, 1]) * 0.08

    t1, t2, t3 = motor_positions[:, 0], motor_positions[:,
                                                        1], motor_positions[:,
                                                                            2]
    l_eff = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(t3))
    t_eff = t2 + t3 / 2
    J = np.zeros((4, 3, 3))
    J[:, 0, 0] = 0
    J[:, 0, 1] = -l_eff * np.cos(t_eff)
    J[:, 0, 2] = l_low * l_up * np.sin(t3) * np.sin(
        t_eff) / l_eff - l_eff * np.cos(t_eff) / 2
    J[:, 1, 0] = -l_hip * np.sin(t1) + l_eff * np.cos(t1) * np.cos(t_eff)
    J[:, 1, 1] = -l_eff * np.sin(t1) * np.sin(t_eff)
    J[:, 1, 2] = -l_low * l_up * np.sin(t1) * np.sin(t3) * np.cos(
        t_eff) / l_eff - l_eff * np.sin(t1) * np.sin(t_eff) / 2
    J[:, 2, 0] = l_hip * np.cos(t1) + l_eff * np.sin(t1) * np.cos(t_eff)
    J[:, 2, 1] = l_eff * np.sin(t_eff) * np.cos(t1)
    J[:, 2, 2] = l_low * l_up * np.sin(t3) * np.cos(t1) * np.cos(
        t_eff) / l_eff + l_eff * np.sin(t_eff) * np.cos(t1) / 2

    flattened_jacobian = np.zeros((12, 12))
    flattened_jacobian[:3, :3] = J[0]
    flattened_jacobian[3:6, 3:6] = J[1]
    flattened_jacobian[6:9, 6:9] = J[2]
    flattened_jacobian[9:12, 9:12] = J[3]
    return flattened_jacobian

  def set_foot_frictions(self, friction_coef):
    del friction_coef  # unused

  def render(self):
    pass

  @property
  def all_foot_jacobian(self):
    return to_torch(self._jacobians[None, :, :], device=self._device)

  @property
  def base_position_world(self):
    return to_torch([self._state_estimator.estimated_position],
                    device=self._device)

  @property
  def base_position(self):
    return to_torch([self._state_estimator.estimated_position],
                    device=self._device)

  @property
  def base_orientation_rpy(self):
    return to_torch([self._raw_state.imu.rpy], device=self._device)

  @property
  def base_orientation_quat(self):
    return self._base_quat_torch

  @property
  def projected_gravity(self):
    return self.base_rot_mat[:, :, 2]

  @property
  def base_rot_mat(self):
    return self._base_rot_mat_torch

  @property
  def base_rot_mat_numpy(self):
    return self._base_rot_mat.copy()

  @property
  def base_rot_mat_t(self):
    return self._base_rot_mat_t_torch

  @property
  def base_rot_mat_zero_yaw(self):
    return self._base_rot_mat_zero_yaw_torch

  @property
  def base_rot_mat_zero_yaw_t(self):
    return self._base_rot_mat_zero_yaw_t_torch

  @property
  def base_velocity_world_frame(self):
    return to_torch(self._state_estimator.estimated_velocity[None, :],
                    device=self._device)

  @property
  def base_velocity_body_frame(self):
    return to_torch(self._base_rot_mat.T.dot(
        self._state_estimator.estimated_velocity)[None, :],
                    device=self._device)

  @property
  def base_velocity_gravity_frame(self):
    return to_torch(self._base_rot_mat_zero_yaw.dot(
        self.base_velocity_body_frame[0])[None, :],
                    device=self._device)

  @property
  def base_angular_velocity_world_frame(self):
    return to_torch(self._base_rot_mat.dot(
        self._state_estimator.angular_velocity)[None, :],
                    device=self._device)

  @property
  def base_angular_velocity_gravity_frame(self):
    return to_torch(self._base_rot_mat_zero_yaw.dot(
        self._state_estimator.angular_velocity)[None, :],
                    device=self._device)

  @property
  def base_angular_velocity_body_frame(self):
    """Smoothed using moving-window filter"""
    return to_torch(self._state_estimator.angular_velocity[None, :],
                    device=self._device)

  @property
  def motor_positions(self):
    return to_torch([[motor.q for motor in self._raw_state.motorState[:12]]],
                    device=self._device)

  @property
  def motor_positions_numpy(self):
    return np.array([motor.q for motor in self._raw_state.motorState[:12]])

  @property
  def motor_velocities(self):
    return to_torch([[motor.dq for motor in self._raw_state.motorState[:12]]],
                    device=self._device)

  @property
  def motor_velocities_numpy(self):
    return np.array([motor.dq for motor in self._raw_state.motorState[:12]])

  @property
  def motor_torques(self):
    return to_torch(
        [[motor.tauEst for motor in self._raw_state.motorState[:12]]],
        device=self._device)

  @property
  def foot_positions_in_base_frame(self):
    return to_torch(self._foot_positions_in_base_frame[None, :, :],
                    device=self._device)

  @property
  def foot_positions_in_base_frame_numpy(self):
    return self._foot_positions_in_base_frame.copy()

  @property
  def foot_positions_in_gravity_frame(self):
    return to_torch(self._foot_positions_in_gravity_frame[None, :, :],
                    device=self._device)

  @property
  def foot_positions_in_gravity_frame_numpy(self):
    return self._foot_positions_in_gravity_frame.copy()

  @property
  def foot_velocities_in_base_frame(self):
    foot_vels = torch.bmm(self.all_foot_jacobian,
                          self.motor_velocities[:, :, None]).squeeze()
    return foot_vels.reshape((self._num_envs, 4, 3))

  @property
  def foot_height(self):
    return torch.where(self.foot_contacts, 0.02, 0.05)

  @property
  def foot_velocities_in_world_frame(self):
    # logging.warning("World-frame foot velocity is not yet implemented.")
    return torch.zeros((self._num_envs, 4, 3))

  @property
  def foot_contacts(self):
    return torch.tensor(self.foot_contacts_numpy[None, :], device=self._device)

  @property
  def foot_contacts_numpy(self):
    return self.foot_contact_forces_numpy > self._contact_force_threshold

  @property
  def calf_contacts(self):
    # Can't detect calf contacts, return default value
    return torch.zeros((self._num_envs, 4), device=self._device)

  @property
  def thigh_contacts(self):
    # Can't detect thigh contacts, return default value
    return torch.zeros((self._num_envs, 4), device=self._device)

  @property
  def has_body_contact(self):
    return to_torch([False], device=self._device)

  @property
  def has_nonfoot_contact(self):
    return to_torch([False], device=self._device)

  @property
  def foot_contact_forces(self):
    return to_torch([self._raw_state.footForce], device=self._device)

  @property
  def foot_contact_forces_numpy(self):
    return np.array(self._raw_state.footForce)

  @property
  def time_since_reset(self):
    return to_torch([time.time() - self._last_reset_time], device=self._device)

  @property
  def time_since_reset_scalar(self):
    return time.time() - self._last_reset_time

  @property
  def mpc_body_height(self):
    return 0.26

  @property
  def raw_state(self):
    return self._raw_state

  @property
  def robot_interface(self):
    return self._robot_interface

  @property
  def state_estimator(self):
    return self._state_estimator

  @property
  def num_envs(self):
    return self._num_envs

  @property
  def num_dof(self):
    return self._num_dof

  @property
  def motor_group(self):
    return self._motors

  @property
  def device(self):
    return self._device

  @property
  def control_timestep(self):
    return self._sim_config.dt * self._sim_config.action_repeat

  @property
  def hip_positions_in_body_frame(self):
    return self._hip_positions_in_body_frame

  @property
  def hip_offset(self):
    """Position of hip offset in base frame, used for IK only."""
    return self._hip_offset
