"""Reproduction of torch_utils from IsaacGym to avoid dependency."""

import torch
import numpy as np


def to_torch(x, dtype=torch.float, device='cuda', requires_grad=False):
  return torch.tensor(x,
                      dtype=dtype,
                      device=device,
                      requires_grad=requires_grad)


@torch.jit.script
def quat_to_rot_mat(q):
  x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  Nq = w * w + x * x + y * y + z * z
  s = 2.0 / Nq
  X, Y, Z = x * s, y * s, z * s
  wX, wY, wZ = w * X, w * Y, w * Z
  xX, xY, xZ = x * X, x * Y, x * Z
  yY, yZ = y * Y, y * Z
  zZ = z * Z

  rotation_matrix = torch.stack([
      torch.stack([1.0 - (yY + zZ), xY - wZ, xZ + wY], dim=-1),
      torch.stack([xY + wZ, 1.0 - (xX + zZ), yZ - wX], dim=-1),
      torch.stack([xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=-1)
  ],
                                dim=-2)

  return rotation_matrix


@torch.jit.script
def copysign(a, b):
  # type: (float, torch.Tensor) -> torch.Tensor
  a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
  return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz_from_quaternion(q):
  qx, qy, qz, qw = 0, 1, 2, 3
  # roll (x-axis rotation)
  sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
  cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
      q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
  roll = torch.atan2(sinr_cosp, cosr_cosp)

  # pitch (y-axis rotation)
  sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
  pitch = torch.where(
      torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

  # yaw (z-axis rotation)
  siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
  cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
      q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
  yaw = torch.atan2(siny_cosp, cosy_cosp)

  return torch.stack(
      (roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)), dim=1)


@torch.jit.script
def quat_mul(a, b):
  assert a.shape == b.shape
  shape = a.shape
  a = a.reshape(-1, 4)
  b = b.reshape(-1, 4)

  x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
  x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
  ww = (z1 + x1) * (x2 + y2)
  yy = (w1 - y1) * (w2 + z2)
  zz = (w1 + y1) * (w2 - z2)
  xx = ww + yy + zz
  qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
  w = qq - ww + (z1 - y1) * (y2 - z2)
  x = qq - xx + (x1 + w1) * (x2 + w2)
  y = qq - yy + (w1 - x1) * (y2 + z2)
  z = qq - zz + (z1 + y1) * (w2 - x2)

  quat = torch.stack([x, y, z, w], dim=-1).view(shape)

  return quat


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
  cy = torch.cos(yaw * 0.5)
  sy = torch.sin(yaw * 0.5)
  cr = torch.cos(roll * 0.5)
  sr = torch.sin(roll * 0.5)
  cp = torch.cos(pitch * 0.5)
  sp = torch.sin(pitch * 0.5)

  qw = cy * cr * cp + sy * sr * sp
  qx = cy * sr * cp - sy * cr * sp
  qy = cy * cr * sp + sy * sr * cp
  qz = sy * cr * cp - cy * sr * sp

  return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def quat_rotate(q, v):
  shape = q.shape
  q_w = q[:, -1]
  q_vec = q[:, :3]
  a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
  b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
  c = q_vec * \
      torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
          shape[0], 3, 1)).squeeze(-1) * 2.0
  return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
  shape = q.shape
  q_w = q[:, -1]
  q_vec = q[:, :3]
  a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
  b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
  c = q_vec * \
      torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
          shape[0], 3, 1)).squeeze(-1) * 2.0
  return a - b + c
