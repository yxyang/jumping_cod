"""Vectorized Go1 robot in Isaac Gym."""
from typing import Any, Sequence, Union

import ml_collections
import torch

from src.robots.robot import Robot
from src.robots.motors import MotorControlMode, MotorGroup, MotorModel
from src.utils.torch_utils import to_torch

_ARRAY = Sequence[float]


@torch.jit.script
def motor_angles_from_foot_positions(foot_local_positions,
                                     hip_offset,
                                     device: str = "cuda"):
  foot_positions_in_hip_frame = foot_local_positions - hip_offset
  l_up = 0.213
  l_low = 0.213
  l_hip = 0.08 * torch.tensor([-1, 1, -1, 1], device=device)

  x = foot_positions_in_hip_frame[:, :, 0]
  y = foot_positions_in_hip_frame[:, :, 1]
  z = foot_positions_in_hip_frame[:, :, 2]
  theta_knee = -torch.arccos(
      torch.clip((x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) /
                 (2 * l_low * l_up), -1, 1))
  l = torch.sqrt(
      torch.clip(l_up**2 + l_low**2 + 2 * l_up * l_low * torch.cos(theta_knee),
                 1e-7, 1))
  theta_hip = torch.arcsin(torch.clip(-x / l, -1, 1)) - theta_knee / 2
  c1 = l_hip * y - l * torch.cos(theta_hip + theta_knee / 2) * z
  s1 = l * torch.cos(theta_hip + theta_knee / 2) * y + l_hip * z
  theta_ab = torch.arctan2(s1, c1)

  # thetas: num_envs x 4
  joint_angles = torch.stack([theta_ab, theta_hip, theta_knee], dim=2)
  return joint_angles.reshape((-1, 12))


class Go1(Robot):
  """Go1 robot in simulation."""
  def __init__(self,
               sim: Any,
               viewer: Any,
               sim_config: ml_collections.ConfigDict,
               num_envs: int,
               init_positions: _ARRAY,
               motor_control_mode: MotorControlMode,
               terrain: Any,
               motor_torque_delay_steps: int = 0,
               camera_config: Union[ml_collections.ConfigDict, None] = None,
               randomize_com: bool = False):

    motors = MotorGroup(device=sim_config.sim_device,
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
                        torque_delay_steps=motor_torque_delay_steps)

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

    super().__init__(sim=sim,
                     viewer=viewer,
                     num_envs=num_envs,
                     init_positions=init_positions,
                     urdf_path="data/go1/urdf/go1.urdf",
                     sim_config=sim_config,
                     motors=motors,
                     feet_names=[
                         "1_FR_foot",
                         "2_FL_foot",
                         "3_RR_foot",
                         "4_RL_foot",
                     ],
                     calf_names=[
                         "1_FR_calf",
                         "2_FL_calf",
                         "3_RR_calf",
                         "4_RL_calf",
                     ],
                     thigh_names=[
                         "1_FR_thigh",
                         "2_FL_thigh",
                         "3_RR_thigh",
                         "4_RL_thigh",
                     ],
                     terrain=terrain,
                     camera_config=camera_config,
                     randomize_com=randomize_com)

  @property
  def hip_positions_in_body_frame(self):
    return self._hip_positions_in_body_frame

  @property
  def hip_offset(self):
    """Position of hip offset in base frame, used for IK only."""
    return self._hip_offset

  def get_motor_angles_from_foot_positions(self, foot_local_positions):
    return motor_angles_from_foot_positions(foot_local_positions,
                                            self.hip_offset,
                                            device=self._device)
