"""Solves the centroidal QP to compute desired foot torques."""
import time
from typing import Any

import numpy as np
import torch
from qpth.qp import QPFunction, QPSolvers
import quadprog

from src.robots.motors import MotorCommand
from src.utils.torch_utils import quat_mul, quat_from_euler_xyz, to_torch, quat_rotate, quat_to_rot_mat


@torch.jit.script
def cross_quad(v1, v2):
  """Assumes v1 is nx3, v2 is nx4x3"""
  v1 = torch.stack([v1, v1, v1, v1], dim=1)
  shape = v1.shape
  v1 = v1.reshape((-1, 3))
  v2 = v2.reshape((-1, 3))
  return torch.cross(v1, v2).reshape((shape[0], shape[1], 3))


@torch.jit.script
def quaternion_to_axis_angle(q):
  angle = 2 * torch.acos(torch.clip(q[:, 3], -0.99999, 0.99999))[:, None]
  norm = torch.clip(torch.linalg.norm(q[:, :3], dim=1), 1e-5, 1)[:, None]
  axis = q[:, :3] / norm
  return axis, angle


@torch.jit.script
def compute_orientation_error(desired_orientation_rpy,
                              base_orientation_quat,
                              device: str = 'cuda'):
  desired_quat = quat_from_euler_xyz(
      desired_orientation_rpy[:, 0], desired_orientation_rpy[:, 1],
      torch.zeros_like(desired_orientation_rpy[:, 2]))
  base_quat_inv = torch.clone(base_orientation_quat)
  base_quat_inv[:, -1] *= -1
  error_quat = quat_mul(desired_quat, base_quat_inv)
  axis, angle = quaternion_to_axis_angle(error_quat)
  angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)
  error_so3 = axis * angle
  return quat_rotate(base_orientation_quat, error_so3)


@torch.jit.script
def compute_desired_acc(
    base_orientation_rpy: torch.Tensor,
    base_position: torch.Tensor,
    base_angular_velocity_body_frame: torch.Tensor,
    base_velocity_body_frame: torch.Tensor,
    desired_base_orientation_rpy: torch.Tensor,
    desired_base_position: torch.Tensor,
    desired_angular_velocity: torch.Tensor,
    desired_linear_velocity: torch.Tensor,
    desired_angular_acceleration: torch.Tensor,
    desired_linear_acceleration: torch.Tensor,
    base_position_kp: torch.Tensor,
    base_position_kd: torch.Tensor,
    base_orientation_kp: torch.Tensor,
    base_orientation_kd: torch.Tensor,
    device: str = "cuda",
):
  base_rpy = base_orientation_rpy
  base_quat = quat_from_euler_xyz(
      base_rpy[:, 0], base_rpy[:, 1],
      torch.zeros_like(base_rpy[:, 0], device=device))
  base_rot_mat_zero_yaw = quat_to_rot_mat(base_quat)
  base_rot_mat_zero_yaw_t = base_rot_mat_zero_yaw.transpose(1, 2)

  lin_pos_error = desired_base_position - base_position
  lin_pos_error[:, :2].zero_()
  lin_vel_error = desired_linear_velocity - (
      base_rot_mat_zero_yaw @ base_velocity_body_frame.unsqueeze(2)).squeeze(2)
  desired_lin_acc_gravity_frame = (base_position_kp * lin_pos_error +
                                   base_position_kd * lin_vel_error +
                                   desired_linear_acceleration)

  ang_pos_error = compute_orientation_error(desired_base_orientation_rpy,
                                            base_quat,
                                            device=device)
  ang_vel_error = desired_angular_velocity - (
      base_rot_mat_zero_yaw
      @ base_angular_velocity_body_frame.unsqueeze(2)).squeeze(2)
  desired_ang_acc_gravity_frame = (base_orientation_kp * ang_pos_error +
                                   base_orientation_kd * ang_vel_error +
                                   desired_angular_acceleration)

  desired_lin_acc_body_frame = (
      base_rot_mat_zero_yaw_t
      @ desired_lin_acc_gravity_frame.unsqueeze(2)).squeeze(2)
  desired_ang_acc_body_frame = (
      base_rot_mat_zero_yaw_t
      @ desired_ang_acc_gravity_frame.unsqueeze(2)).squeeze(2)

  return torch.cat((desired_lin_acc_body_frame, desired_ang_acc_body_frame),
                   dim=1)


@torch.jit.script
def convert_to_skew_symmetric_batch(foot_positions):
  """
  Converts foot positions (nx4x3) into skew-symmetric ones (nx3x12)
  """
  n = foot_positions.shape[0]
  x = foot_positions[:, :, 0]  # - 0.023
  y = foot_positions[:, :, 1]
  z = foot_positions[:, :, 2]
  zero = torch.zeros_like(x)
  skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).reshape(
      (n, 3, 3, 4))
  return torch.concatenate(
      [skew[:, :, :, 0], skew[:, :, :, 1], skew[:, :, :, 2], skew[:, :, :, 3]],
      dim=2)


@torch.jit.script
def construct_mass_mat(foot_positions,
                       foot_contact_state,
                       inv_mass,
                       inv_inertia,
                       device: str = 'cuda',
                       mask_noncontact_legs: bool = True):
  num_envs = foot_positions.shape[0]
  mass_mat = torch.zeros((num_envs, 6, 12), device=device)
  # Construct mass matrix
  inv_mass_concat = torch.concatenate([inv_mass] * 4, dim=1)
  mass_mat[:, :3] = inv_mass_concat[None, :, :]
  px = convert_to_skew_symmetric_batch(foot_positions)
  mass_mat[:, 3:6] = torch.matmul(inv_inertia, px)
  # Mark out non-contact legs
  if mask_noncontact_legs:
    non_contact_indices = torch.nonzero(torch.logical_not(foot_contact_state))
    env_id, leg_id = non_contact_indices[:, 0], non_contact_indices[:, 1]
    mass_mat[env_id, :, leg_id * 3] = 0
    mass_mat[env_id, :, leg_id * 3 + 1] = 0
    mass_mat[env_id, :, leg_id * 3 + 2] = 0
  return mass_mat


@torch.jit.script
def solve_grf(mass_mat,
              desired_acc,
              base_rot_mat_t,
              Wq,
              Wf: float,
              foot_friction_coef: float,
              clip_grf: bool,
              foot_contact_state,
              device: str = 'cuda'):
  num_envs = mass_mat.shape[0]
  g = torch.zeros((num_envs, 6), device=device)
  g[:, 2] = 9.8

  g[:, :3] = torch.matmul(base_rot_mat_t, g[:, :3, None])[:, :, 0]
  Q = torch.zeros((num_envs, 6, 6), device=device) + Wq[None, :]
  Wf_mat = torch.eye(12, device=device) * Wf
  R = torch.zeros((num_envs, 12, 12), device=device) + Wf_mat[None, :]

  quad_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q),
                        mass_mat) + R
  linear_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q),
                          (g + desired_acc)[:, :, None])[:, :, 0]
  # grf = torch.linalg.solve(quad_term, linear_term)
  grf = torch.bmm(torch.linalg.pinv(quad_term), linear_term[:, :, None])[:, :,
                                                                         0]

  base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)
  grf = grf.reshape((-1, 4, 3))
  grf_world = torch.transpose(
      torch.bmm(base_rot_mat, torch.transpose(grf, 1, 2)), 1, 2)
  clipped_motor = torch.logical_or(
      grf_world[:, :, 2] < 10,
      grf_world[:, :, 2] > 130) * torch.logical_not(foot_contact_state)
  if clip_grf:
    grf_world[:, :, 2] = grf_world[:, :, 2].clip(min=10, max=130)
    grf_world[:, :, 2] *= foot_contact_state

  friction_force = torch.norm(grf_world[:, :, :2], dim=2) + 0.001
  max_friction_force = foot_friction_coef * grf_world[:, :, 2].clip(min=0)
  multiplier = torch.where(friction_force < max_friction_force, 1,
                           max_friction_force / friction_force)

  clipped_grf = torch.logical_or(multiplier < 1, clipped_motor)

  if clip_grf:
    grf_world[:, :, :2] *= multiplier[:, :, None]
  grf = torch.transpose(
      torch.bmm(base_rot_mat_t, torch.transpose(grf_world, 1, 2)), 1, 2)
  grf = grf.reshape((-1, 12))

  # Convert to motor torques
  solved_acc = torch.bmm(mass_mat, grf[:, :, None])[:, :, 0] - g
  qp_cost = torch.bmm(
      torch.bmm((solved_acc - desired_acc)[:, :, None].transpose(1, 2), Q),
      (solved_acc - desired_acc)[:, :, None])[:, 0, 0]
  return grf, solved_acc, qp_cost, torch.sum(clipped_grf, dim=1)


# @torch.jit.script
def solve_grf_qpth(mass_mat,
                   desired_acc,
                   base_rot_mat_t,
                   Wq,
                   Wf: float,
                   foot_friction_coef: float,
                   clip_grf: bool,
                   foot_contact_state,
                   device: str = 'cuda'):
  num_envs = mass_mat.shape[0]
  g = torch.zeros((num_envs, 6), device=device)
  g[:, 2] = 9.8

  g[:, :3] = torch.matmul(base_rot_mat_t, g[:, :3, None])[:, :, 0]
  Q = torch.zeros((num_envs, 6, 6), device=device) + Wq[None, :]
  Wf_mat = torch.eye(12, device=device) * Wf
  R = torch.zeros((num_envs, 12, 12), device=device) + Wf_mat[None, :]

  quad_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q),
                        mass_mat) + R
  linear_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q),
                          (g + desired_acc)[:, :, None])[:, :, 0]

  G = torch.zeros((mass_mat.shape[0], 24, 12), device=device)
  h = torch.zeros((mass_mat.shape[0], 24), device=device) + 1e-3
  base_rot_mat = torch.transpose(base_rot_mat_t, 1, 2)
  for leg_id in range(4):
    G[:, leg_id * 2, leg_id * 3 + 2] = 1
    G[:, leg_id * 2 + 1, leg_id * 3 + 2] = -1

    row_id, col_id = 8 + leg_id * 4, leg_id * 3
    G[:, row_id, col_id] = 1
    G[:, row_id, col_id + 2] = -foot_friction_coef

    G[:, row_id + 1, col_id] = -1
    G[:, row_id + 1, col_id + 2] = -foot_friction_coef

    G[:, row_id + 2, col_id + 1] = 1
    G[:, row_id + 2, col_id + 2] = -foot_friction_coef

    G[:, row_id + 3, col_id + 1] = -1
    G[:, row_id + 3, col_id + 2] = -foot_friction_coef
    G[:, leg_id * 2:leg_id * 2 + 2, col_id:col_id + 3] = torch.bmm(
        G[:, leg_id * 2:leg_id * 2 + 2, col_id:col_id + 3], base_rot_mat)
    G[:, row_id:row_id + 4, col_id:col_id + 3] = torch.bmm(
        G[:, row_id:row_id + 4, col_id:col_id + 3], base_rot_mat)

  contact_ids = foot_contact_state.nonzero()

  h[contact_ids[:, 0], contact_ids[:, 1] * 2] = 130
  h[contact_ids[:, 0], contact_ids[:, 1] * 2 + 1] = -10
  e = torch.autograd.Variable(torch.Tensor())

  qf = QPFunction(verbose=-1,
                  check_Q_spd=False,
                  eps=1e-3,
                  solver=QPSolvers.PDIPM_BATCHED)
  grf = qf(quad_term.double(), -linear_term.double(), G.double(), h.double(),
           e, e).float()
  # print(grf)
  # ans = input("Any Key...")
  # if ans in ["Y", "y"]:
  #   import pdb
  #   pdb.set_trace()
  solved_acc = torch.bmm(mass_mat, grf[:, :, None])[:, :, 0] - g
  qp_cost = torch.bmm(
      torch.bmm((solved_acc - desired_acc)[:, :, None].transpose(1, 2), Q),
      (solved_acc - desired_acc)[:, :, None])[:, 0, 0]

  return grf, solved_acc, qp_cost, torch.zeros(mass_mat.shape[0],
                                               device=device)


def solve_grf_quadprog(mass_mat,
                       desired_acc,
                       base_rot_mat_t,
                       Wq,
                       Wf,
                       foot_friction_coef: float,
                       clip_grf: bool,
                       foot_contact_state,
                       device: str = 'cuda'):
  num_envs = mass_mat.shape[0]
  if num_envs != 1:
    raise RuntimeError("Not Supported!")

  mass_mat = np.array(mass_mat[0])
  desired_acc = np.array(desired_acc[0])
  base_rot_mat_t = np.array(base_rot_mat_t[0])
  Wq = np.array(Wq)
  foot_contact_state = np.array(foot_contact_state[0])

  g = np.zeros(6)
  g[2] = 9.8
  g[:3] = base_rot_mat_t.dot(g[:3])  # Convert to body frame
  Q = np.zeros((6, 6)) + Wq
  R = np.eye(12) * Wf.item()
  quad_term = mass_mat.T.dot(Q).dot(mass_mat) + R
  linear_term = mass_mat.T.dot(Q).dot(g + desired_acc)

  G = np.zeros((24, 12))
  h = np.zeros(24) + 1e-3
  base_rot_mat = base_rot_mat_t.T
  for leg_id in range(4):
    G[leg_id * 2, leg_id * 3 + 2] = 1
    G[leg_id * 2 + 1, leg_id * 3 + 2] = -1

    row_id, col_id = 8 + leg_id * 4, leg_id * 3
    G[row_id, col_id] = 1
    G[row_id, col_id + 2] = -foot_friction_coef

    G[row_id + 1, col_id] = -1
    G[row_id + 1, col_id + 2] = -foot_friction_coef

    G[row_id + 2, col_id + 1] = 1
    G[row_id + 2, col_id + 2] = -foot_friction_coef

    G[row_id + 3, col_id + 1] = -1
    G[row_id + 3, col_id + 2] = -foot_friction_coef
    G[leg_id * 2:leg_id * 2 + 2,
      col_id:col_id + 3] = G[leg_id * 2:leg_id * 2 + 2,
                             col_id:col_id + 3].dot(base_rot_mat)
    G[row_id:row_id + 4,
      col_id:col_id + 3] = G[row_id:row_id + 4,
                             col_id:col_id + 3].dot(base_rot_mat)

  contact_ids = foot_contact_state.nonzero()[0]

  h[contact_ids * 2] = 130
  h[contact_ids * 2 + 1] = -10

  grf = quadprog.solve_qp(quad_term, linear_term, -G.T, -h)[0]
  solved_acc = (mass_mat.dot(grf) - g)
  qp_cost = (solved_acc - desired_acc).T.dot(Q).dot(solved_acc - desired_acc)
  grf = to_torch(grf[None, :], device=device)
  solved_acc = to_torch(solved_acc[None, :], device=device)
  qp_cost = to_torch([qp_cost], device=device)

  return grf, solved_acc, qp_cost, torch.zeros(num_envs, device=device)


class QPTorqueOptimizer:
  """Centroidal QP controller to optimize for joint torques.

  Odometry = Zero-yaw frame / Roomba frame.
  """
  def __init__(self,
               robot: Any,
               gait_generator: Any,
               base_position_kp=np.array([50., 50., 50]),
               base_position_kd=np.array([1., 1., 1.]),
               base_orientation_kp=np.array([50., 50., 0.]),
               base_orientation_kd=np.array([1., 1., 10.]),
               weight_ddq=np.diag([1., 1., 10., 10., 10., 1.]),
               weight_grf=1e-4,
               body_mass=14.076,
               body_inertia=np.array([0.14, 0.35, 0.35]) * 1.5,
               desired_body_height=0.26,
               foot_friction_coef=0.7,
               clip_grf=False,
               swing_foot_limit=None,
               use_full_qp=False):
    """Initializes the controller with desired weights and gains."""
    self._robot = robot
    self._gait_generator = gait_generator
    self._device = self._robot._device
    self._num_envs = self._robot.num_envs
    self._clip_grf = clip_grf
    self._use_full_qp = use_full_qp

    self._base_orientation_kp = to_torch(base_orientation_kp,
                                         device=self._device)
    self._base_orientation_kp = torch.stack([self._base_orientation_kp] *
                                            self._num_envs,
                                            dim=0)
    self._base_orientation_kd = to_torch(base_orientation_kd,
                                         device=self._device)
    self._base_orientation_kd = torch.stack([self._base_orientation_kd] *
                                            self._num_envs,
                                            dim=0)
    self._base_position_kp = to_torch(base_position_kp, device=self._device)
    self._base_position_kp = torch.stack([self._base_position_kp] *
                                         self._num_envs,
                                         dim=0)
    self._base_position_kd = to_torch(base_position_kd, device=self._device)
    self._base_position_kd = torch.stack([self._base_position_kd] *
                                         self._num_envs,
                                         dim=0)
    self._desired_base_orientation_rpy = torch.zeros((self._num_envs, 3),
                                                     device=self._device)
    self._desired_base_position = torch.zeros((self._num_envs, 3),
                                              device=self._device)
    self._desired_base_position[:, 2] = desired_body_height
    self._desired_linear_velocity = torch.zeros((self._num_envs, 3),
                                                device=self._device)
    self._desired_angular_velocity = torch.zeros((self._num_envs, 3),
                                                 device=self._device)
    self._desired_linear_acceleration = torch.zeros((self._num_envs, 3),
                                                    device=self._device)
    self._desired_angular_acceleration = torch.zeros((self._num_envs, 3),
                                                     device=self._device)
    self._Wq = to_torch(weight_ddq, device=self._device, dtype=torch.float32)
    self._Wf = to_torch(weight_grf, device=self._device)
    self._foot_friction_coef = foot_friction_coef
    self._inv_mass = torch.eye(3, device=self._device) / body_mass
    self._inv_inertia = torch.linalg.inv(
        torch.diag(to_torch(body_inertia, device=self._device)))

    if swing_foot_limit is None:
      self._swing_foot_limit = (-0.4, -0.1)
    else:
      self._swing_foot_limit = swing_foot_limit

    self._last_leg_state = self._gait_generator.desired_contact_state
    self._last_timestamp = self._robot.time_since_reset

  def reset(self) -> None:
    pass

  def reset_idx(self, env_ids) -> None:
    pass

  def update(self) -> None:
    # compute dt and stance foot odometry position
    pass

  def _solve_joint_torques(self, foot_contact_state, desired_com_ddq):
    """Solves centroidal QP to find desired joint torques."""
    self._mass_mat = construct_mass_mat(
        self._robot.foot_positions_in_base_frame,
        foot_contact_state,
        self._inv_mass,
        self._inv_inertia,
        mask_noncontact_legs=not self._use_full_qp,
        device=self._device)

    # Solve QP
    if self._use_full_qp:
      if self._num_envs == 1:
        grf, solved_acc, qp_cost, num_clips = solve_grf_quadprog(
            self._mass_mat,
            desired_com_ddq,
            self._robot.base_rot_mat_zero_yaw_t,
            self._Wq,
            self._Wf,
            self._foot_friction_coef,
            self._clip_grf,
            foot_contact_state,
            device=self._device)
      else:
        grf, solved_acc, qp_cost, num_clips = solve_grf_qpth(
            self._mass_mat,
            desired_com_ddq,
            self._robot.base_rot_mat_zero_yaw_t,
            self._Wq,
            self._Wf,
            self._foot_friction_coef,
            self._clip_grf,
            foot_contact_state,
            device=self._device)
    else:
      grf, solved_acc, qp_cost, num_clips = solve_grf(
          self._mass_mat,
          desired_com_ddq,
          self._robot.base_rot_mat_zero_yaw_t,
          self._Wq,
          self._Wf,
          self._foot_friction_coef,
          self._clip_grf,
          foot_contact_state,
          device=self._device)

    all_foot_jacobian = self._robot.all_foot_jacobian
    motor_torques = -torch.bmm(grf[:, None, :], all_foot_jacobian)[:, 0]
    return motor_torques, solved_acc, grf, qp_cost, num_clips

  def compute_joint_command(self, foot_contact_state: torch.Tensor,
                            desired_base_orientation_rpy: torch.Tensor,
                            desired_base_position: torch.Tensor,
                            desired_swing_foot_position: torch.Tensor,
                            desired_angular_velocity: torch.Tensor,
                            desired_linear_velocity: torch.Tensor,
                            desired_foot_velocity: torch.Tensor,
                            desired_angular_acceleration: torch.Tensor,
                            desired_linear_acceleration: torch.Tensor,
                            desired_foot_acceleration: torch.Tensor):

    # desired_base_position = torch.clone(desired_base_position).clip(
    #     self._robot.base_position - 0.05, self._robot.base_position + 0.05)
    # desired_base_orientation_rpy = torch.clone(
    #     desired_base_orientation_rpy).clip(
    #         self._robot.base_orientation_rpy - 0.2,
    #         self._robot.base_orientation_rpy + 0.2)

    # Position command
    # desired_quat = quat_from_euler_xyz(
    #     desired_base_orientation_rpy[:, 0], desired_base_orientation_rpy[:, 1],
    #     torch.zeros_like(desired_base_orientation_rpy[:, 0]))
    # desired_rot_mat = quat_to_rot_mat(desired_quat)
    # desired_stance_foot_position = torch.clone(
    #     self._foot_position_odometry_frame)
    # desired_stance_foot_position[..., 2] -= desired_base_position[:, None, 2]
    # desired_stance_foot_position = torch.matmul(
    #     desired_rot_mat.transpose(1, 2),
    #     desired_stance_foot_position.transpose(1, 2)).transpose(1, 2)

    desired_swing_foot_position = torch.bmm(
        self._robot.base_rot_mat_t,
        desired_swing_foot_position.transpose(1, 2)).transpose(1, 2)
    desired_swing_foot_position[:, :, 2] = torch.clip(
        desired_swing_foot_position[:, :, 2],
        min=self._swing_foot_limit[0],
        max=self._swing_foot_limit[1])
    # desired_foot_position = torch.where(
    #     torch.tile(self._gait_generator.desired_contact_state[:, :, None],
    #                [1, 1, 3]), desired_stance_foot_position,
    #     desired_swing_foot_position)
    desired_foot_position = torch.where(
        torch.tile(self._gait_generator.desired_contact_state[:, :, None],
                   [1, 1, 3]), self._robot.foot_positions_in_base_frame,
        desired_swing_foot_position)

    desired_position = self._robot.get_motor_angles_from_foot_positions(
        desired_foot_position)

    # if (desired_position < self._robot.motor_group.min_positions).any():
    #   import pdb
    #   pdb.set_trace()
    # if (desired_position > self._robot.motor_group.max_positions).any():
    #   import pdb
    #   pdb.set_trace()

    desired_position = desired_position.clip(
        min=self._robot.motor_group.min_positions + 1e-3,
        max=self._robot.motor_group.max_positions - 1e-3)
    desired_position = desired_position.clip(
        min=self._robot.motor_positions - 0.5,
        max=self._robot.motor_positions + 0.5)

    # Velocity Command
    # foot_position_body_frame = desired_foot_position
    # desired_linear_velocity_fb = torch.clip(
    #     desired_linear_velocity, self._robot.base_velocity_gravity_frame - 100.5,
    #     self._robot.base_angular_velocity_gravity_frame + 100.5)
    desired_base_velocity_body_frame = torch.bmm(
        self._robot.base_rot_mat_zero_yaw_t,
        desired_linear_velocity[:, :, None])[:, :, 0]
    # desired_base_velocity_body_frame = (
    #     0. * self._robot.base_velocity_body_frame +
    #     1. * desired_base_velocity_body_frame)

    # desired_base_velocity_body_frame = torch.bmm(
    #     desired_rot_mat.transpose(1, 2),
    #     desired_linear_velocity[:, :, None])[:, :, 0]
    # desired_angular_velocity_fb = torch.clip(
    #     desired_angular_velocity,
    #     self._robot.base_angular_velocity_gravity_frame - 100,
    #     self._robot.base_angular_velocity_gravity_frame + 100)
    desired_angular_velocity_body_frame = torch.bmm(
        self._robot.base_rot_mat_zero_yaw_t,
        desired_angular_velocity[:, :, None])[:, :, 0]
    # desired_angular_velocity_body_frame = (
    #     0. * self._robot.base_angular_velocity_body_frame +
    #     1. * desired_angular_velocity_body_frame)

    desired_foot_velocity = -desired_base_velocity_body_frame[:, None, :] - cross_quad(
        desired_angular_velocity_body_frame,
        self._robot.foot_positions_in_base_frame)

    desired_stance_velocity = torch.bmm(
        torch.linalg.pinv(self._robot.all_foot_jacobian),
        desired_foot_velocity.reshape((self._num_envs, 12, 1)))[:, :, 0]
    contact_state_expanded = foot_contact_state.repeat_interleave(3, dim=1)

    desired_velocity = torch.where(
        contact_state_expanded, desired_stance_velocity,
        torch.zeros_like(self._robot.motor_velocities))

    # Torque Command
    desired_acc_body_frame = compute_desired_acc(
        self._robot.base_orientation_rpy,
        self._robot.base_position,
        self._robot.base_angular_velocity_body_frame,
        self._robot.base_velocity_body_frame,
        desired_base_orientation_rpy,
        desired_base_position,
        desired_angular_velocity,
        desired_linear_velocity,
        desired_angular_acceleration,
        desired_linear_acceleration,
        self._base_position_kp,
        self._base_position_kd,
        self._base_orientation_kp,
        self._base_orientation_kd,
        device=self._device)
    desired_acc_body_frame = torch.clip(
        desired_acc_body_frame,
        to_torch([-30, -30, -10, -50, -50, -20], device=self._device),
        to_torch([30, 30, 30, 50, 50, 20], device=self._device))
    motor_torques, solved_acc, grf, qp_cost, num_clips = self._solve_joint_torques(
        foot_contact_state, desired_acc_body_frame)
    desired_torque = torch.where(contact_state_expanded, motor_torques,
                                 torch.zeros_like(motor_torques))
    desired_torque = torch.clip(desired_torque,
                                max=self._robot.motor_group.max_torques,
                                min=self._robot.motor_group.min_torques)
    # print(self._robot.time_since_reset)
    # print(f"QP Cost: {qp_cost}, num_clips: {num_clips}")
    # print(f"Desired position: {desired_foot_position}")
    # print(f"Curr Position: {self._robot.foot_positions_in_base_frame}")
    # print("Contact: {}".format(foot_contact_state))
    # print(f"Base pos: {self._robot.base_position}")
    # print("Desired pos: {}".format(desired_base_position))
    # print(f"Base RPY: {self._robot.base_orientation_rpy}")
    # print(f"Stance foot velocity: {desired_foot_velocity}")
    # print(f"Desired RPY: {self._desired_base_orientation_rpy}")
    # print(f"Base Ang Vel: {self._robot.base_angular_velocity_body_frame}")
    # print("Current vel: {}".format(self._robot.base_velocity_body_frame))
    # print("Desired vel: {}".format(desired_linear_velocity))
    # print(f"GRF: {grf.reshape((4, 3))}")
    # print("Desired acc: {}".format(desired_acc_body_frame))
    # print("Desired motor vel: {}".format(desired_velocity))
    # print("Motor Vel: {}".format(self._robot.motor_velocities))
    # print("Feedback Torque: {}".format(
    #     (desired_velocity - self._robot.motor_velocities) *
    #     torch.where(contact_state_expanded, 1, 1)))
    # print(f"FF torque: {desired_torque}")
    # total_torque = (desired_torque +
    #                 (desired_velocity - self._robot.motor_velocities) *
    #                 torch.where(contact_state_expanded, 1, 0))
    # total_grf = -torch.bmm(
    #     torch.linalg.pinv(self._robot.all_foot_jacobian).transpose(1, 2),
    #     total_torque[:, :, None])[:, :, 0]
    # g = torch.zeros((self._num_envs, 6), device=self._device)
    # g[:, 2] = 9.8
    # g[:, :3] = torch.matmul(self._robot.base_rot_mat_zero_yaw_t,
    #                         g[:, :3, None])[:, :, 0]
    # total_acc = torch.bmm(self._mass_mat, total_grf[:, :, None])[:, :, 0] - g
    # total_acc[:, :3] = torch.matmul(self._robot.base_rot_mat_zero_yaw,
    #                          total_acc[:, :3, None])[:, :, 0]
    # total_acc[:, 3:6] = torch.matmul(self._robot.base_rot_mat_zero_yaw,
    #                          total_acc[:, 3:6, None])[:, :, 0]
    # solved_acc[:, :3] = torch.matmul(self._robot.base_rot_mat_zero_yaw,
    #                           solved_acc[:, :3, None])[:, :, 0]
    # solved_acc[:, 3:6] = torch.matmul(self._robot.base_rot_mat_zero_yaw,
    #                           solved_acc[:, 3:6, None])[:, :, 0]
    # print("Combined Acc Gravity Frame: {}".format(total_acc))
    # print("Solved acc gravity frame: {}".format(solved_acc))
    # print("Base Vel: {}".format(self._robot.base_velocity_gravity_frame))

    # ans = input("Any Key...")
    # if ans in ["Y", "y", "Yes", "yes"]:
    #   import pdb
    #   pdb.set_trace()

    return MotorCommand(
        desired_position=desired_position,
        kp=torch.where(contact_state_expanded, 10, 30),
        desired_velocity=desired_velocity,
        kd=torch.where(contact_state_expanded, 1., 1),
        desired_extra_torque=desired_torque
    ), desired_acc_body_frame, solved_acc, qp_cost, num_clips

  def get_action(self, foot_contact_state: torch.Tensor,
                 swing_foot_position: torch.Tensor):
    """Computes motor actions."""
    return self.compute_joint_command(
        foot_contact_state=foot_contact_state,
        desired_base_orientation_rpy=self._desired_base_orientation_rpy,
        desired_base_position=self._desired_base_position,
        desired_swing_foot_position=swing_foot_position,
        desired_angular_velocity=self._desired_angular_velocity,
        desired_linear_velocity=self._desired_linear_velocity,
        desired_foot_velocity=torch.zeros(12),
        desired_angular_acceleration=self._desired_angular_acceleration,
        desired_linear_acceleration=self._desired_linear_acceleration,
        desired_foot_acceleration=torch.zeros(12))

  @property
  def desired_base_position(self) -> torch.Tensor:
    return self._desired_base_position

  @desired_base_position.setter
  def desired_base_position(self, base_position: float):
    self._desired_base_position = to_torch(base_position, device=self._device)

  @property
  def desired_base_orientation_rpy(self) -> torch.Tensor:
    return self._desired_base_orientation_rpy

  @desired_base_orientation_rpy.setter
  def desired_base_orientation_rpy(self, orientation_rpy: torch.Tensor):
    self._desired_base_orientation_rpy = to_torch(orientation_rpy,
                                                  device=self._device)

  @property
  def desired_linear_velocity(self) -> torch.Tensor:
    return self._desired_linear_velocity

  @desired_linear_velocity.setter
  def desired_linear_velocity(self, desired_linear_velocity: torch.Tensor):
    self._desired_linear_velocity = to_torch(desired_linear_velocity,
                                             device=self._device)

  @property
  def desired_angular_velocity(self) -> torch.Tensor:
    return self._desired_angular_velocity

  @desired_angular_velocity.setter
  def desired_angular_velocity(self, desired_angular_velocity: torch.Tensor):
    self._desired_angular_velocity = to_torch(desired_angular_velocity,
                                              device=self._device)

  @property
  def desired_linear_acceleration(self):
    return self._desired_linear_acceleration

  @desired_linear_acceleration.setter
  def desired_linear_acceleration(self,
                                  desired_linear_acceleration: torch.Tensor):
    self._desired_linear_acceleration = to_torch(desired_linear_acceleration,
                                                 device=self._device)

  @property
  def desired_angular_acceleration(self):
    return self._desired_angular_acceleration

  @desired_angular_acceleration.setter
  def desired_angular_acceleration(self,
                                   desired_angular_acceleration: torch.Tensor):
    self._desired_angular_acceleration = to_torch(desired_angular_acceleration,
                                                  device=self._device)
