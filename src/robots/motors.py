"""Implements motor models for different motor control modes."""
from collections import deque

from dataclasses import dataclass
import enum
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch

from src.utils.torch_utils import to_torch

_ARRAY = Sequence[float]
_FloatOrArray = Union[float, _ARRAY]

TORQUE_INPUT_TABLE = np.array([
    -35, -30, -25, -20, -15, -10, -5, -3, 3, 5, 10, 15, 20, 25, 30, 35
]) / 6.33
TORQUE_OUTPUT_TABLE = np.array([
    -25.68, -24.01, -20.34, -17.23, -14.12, -9.78, -5.00, -3.00, 3.00, 5.00,
    9.78, 14.12, 17.23, 20.34, 24.01, 25.68
]) / 6.33


class MotorControlMode(enum.Enum):
  POSITION = 0
  HYBRID = 1


@dataclass
class MotorCommand:
  desired_position: torch.Tensor = torch.zeros(12)
  kp: torch.Tensor = torch.zeros(12)
  desired_velocity: torch.Tensor = torch.zeros(12)
  kd: torch.Tensor = torch.zeros(12)
  desired_extra_torque: torch.Tensor = torch.zeros(12)


def get_torque_interpolation_function(input_table, output_table):
  @torch.jit.script
  def interp(x, input_table=input_table, output_table=output_table):
    # Use torch.searchsorted to find the insertion points
    idx = torch.searchsorted(input_table, x)
    idx = torch.clamp(idx, 1, len(input_table) - 1)

    # Get the x values at the index and the index before
    x0 = input_table[idx - 1]
    x1 = input_table[idx]

    # Get the corresponding y values
    y0 = output_table[idx - 1]
    y1 = output_table[idx]

    # Compute the slopes and intercepts for interpolation
    # Protect against division by zero
    slope = (y1 - y0) / (x1 - x0 + 1e-6)

    # Calculate the interpolated values
    y = y0 + slope * (x - x0)
    return y

  return interp


class MotorModel:
  """Implements a simple DC motor model for simulation.
    To accurately model the motor behaviors, the `MotorGroup` class converts
    all motor commands into torques, which is sent directly to the simulator.
    Each `MotorModel` describes a characteristics of a particular motor.
    NOTE: Until functionality is added to MotorModel, it is effectively
    equivalent to a `dataclass`.
    """

  # TODO(yxyang): Complete documentation of motors with description of units
  # (e.g. rads/s etc.)

  def __init__(
      self,
      name: Optional[str] = None,
      motor_control_mode: MotorControlMode = MotorControlMode.POSITION,
      init_position: float = 0.0,
      min_position: float = 0.0,
      max_position: float = 0.0,
      min_velocity: float = 0.0,
      max_velocity: float = 0.0,
      min_torque: float = 0.0,
      max_torque: float = 0.0,
      kp: float = 0.0,
      kd: float = 0.0,
      reduction_ratio: float = 1.0) -> None:

    self._name = name
    self._motor_control_mode = motor_control_mode
    self._init_position = init_position
    self._min_position = min_position
    self._max_position = max_position
    self._min_velocity = min_velocity
    self._max_velocity = max_velocity
    self._min_torque = min_torque
    self._max_torque = max_torque
    self._kp = kp
    self._kd = kd
    self._reduction_ratio = reduction_ratio


class MotorGroup:
  """Models the behavior of a group of motors."""
  def __init__(self,
               device: str,
               num_envs: int,
               motors: Tuple[MotorModel, ...] = (),
               torque_delay_steps: int = 0):

    self._motors = motors
    self._num_envs = num_envs
    self._num_motors = len(motors)
    self._motor_control_mode = motors[0]._motor_control_mode
    self._device = device
    self._strength_ratios = torch.ones((self._num_envs, self._num_motors),
                                       device=device)
    self._init_positions = to_torch([motor._init_position for motor in motors],
                                    device=device)
    self._min_positions = to_torch([motor._min_position for motor in motors],
                                   device=device)
    self._max_positions = to_torch([motor._max_position for motor in motors],
                                   device=device)
    self._min_velocities = to_torch([motor._min_velocity for motor in motors],
                                    device=device)
    self._max_velocities = to_torch([motor._max_velocity for motor in motors],
                                    device=device)
    self._min_torques = to_torch([motor._min_torque for motor in motors],
                                 device=device)
    self._max_torques = to_torch([motor._max_torque for motor in motors],
                                 device=device)
    self._kps = to_torch([motor._kp for motor in motors], device=device)
    self._kds = to_torch([motor._kd for motor in motors], device=device)
    self._reduction_ratios = to_torch(
        [motor._reduction_ratio for motor in motors], device=device)
    self._torque_history = deque(maxlen=torque_delay_steps + 1)
    self._torque_history.append(
        torch.zeros((self._num_envs, self._num_motors), device=self._device))
    self._torque_output = torch.zeros((self._num_envs, self._num_motors),
                                      device=self._device)
    self._true_motor_torque = torch.zeros((self._num_envs, self._num_motors),
                                          device=self._device)

    self._torque_interpolation_function = get_torque_interpolation_function(
        input_table=to_torch(TORQUE_INPUT_TABLE, device=self._device),
        output_table=to_torch(TORQUE_OUTPUT_TABLE, device=self._device))

  def _clip_torques(self, desired_torque: _ARRAY,
                    current_motor_velocity: _ARRAY):
    torque_ub = torch.where(
        current_motor_velocity < 0, self._max_torques,
        self._max_torques *
        (1 - current_motor_velocity / self._max_velocities))
    torque_lb = torch.where(
        current_motor_velocity < 0,
        self._min_torques *
        (1 - current_motor_velocity / self._min_velocities), self._min_torques)

    return torch.clip(desired_torque, torque_lb, torque_ub)

  def convert_to_torque(
      self,
      command: MotorCommand,
      current_position: _ARRAY,
      current_velocity: _ARRAY,
      motor_control_mode: Optional[MotorControlMode] = None,
  ):
    """Converts the given motor command into motor torques."""
    motor_control_mode = motor_control_mode or self._motor_control_mode
    if motor_control_mode == MotorControlMode.POSITION:
      desired_position = command.desired_position
      kp = self._kps
      desired_velocity = torch.zeros((self._num_envs, self._num_motors),
                                     device=self._device)
      kd = self._kds
    else:  # HYBRID case
      desired_position = command.desired_position
      kp = command.kp
      desired_velocity = command.desired_velocity
      kd = command.kd
      self._torque_history.append(command.desired_extra_torque)
      self._torque_output = 0 * self._torque_output + 1. * self._torque_history[
          0]

    sensed_torque = (kp * (desired_position - current_position) + kd *
                     (desired_velocity - current_velocity) +
                     self._torque_output)

    sensed_torque = self._clip_torques(sensed_torque, current_velocity)
    applied_torque = self._torque_interpolation_function(
        sensed_torque / self._reduction_ratios) * self._reduction_ratios
    applied_torque *= self._strength_ratios
    return applied_torque, sensed_torque

  @property
  def motor_control_mode(self):
    return self._motor_control_mode

  @property
  def kps(self):
    return self._kps

  @kps.setter
  def kps(self, value: _FloatOrArray):
    self._kps = torch.ones(self._num_motors) * value

  @property
  def kds(self):
    return self._kds

  @kds.setter
  def kds(self, value: _FloatOrArray):
    self._kds = torch.ones(self._num_motors) * value

  @property
  def strength_ratios(self):
    return self._strength_ratios

  @strength_ratios.setter
  def strength_ratios(self, value: _FloatOrArray):
    self._strength_ratios = torch.ones(
        self._num_motors, device=self._device) * to_torch(value,
                                                          device=self._device)

  @property
  def init_positions(self):
    return self._init_positions

  @init_positions.setter
  def init_positions(self, value: _ARRAY):
    self._init_positions = value

  @property
  def num_motors(self):
    return self._num_motors

  @property
  def min_positions(self):
    return self._min_positions

  @property
  def max_positions(self):
    return self._max_positions

  @property
  def min_torques(self):
    return self._min_torques

  @property
  def max_torques(self):
    return self._max_torques

  @property
  def reduction_ratios(self):
    return self._reduction_ratios
