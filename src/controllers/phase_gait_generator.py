"""Computes leg states based on sinusoids and phase offsets."""
import copy
from typing import Any

from ml_collections import ConfigDict
import torch

from src.utils.torch_utils import to_torch


class PhaseGaitGenerator:
  """Computes desired gait based on leg phases."""
  def __init__(self, robot: Any, gait_config: ConfigDict):
    """Initializes the gait generator.
    Each gait is parameterized by 3 set of parameters:
      The _stepping frequency_: controls how fast the gait progresses.
      The _offset_: a 4-dim vector representing the offset from a standard
        gait cycle. In a standard gait cycle, each gait cycle starts in stance
        and ends in swing.
      The _swing ratio_: the percentage of air phase in each gait.
    """
    self._robot = robot
    self._num_envs = self._robot.num_envs
    self._device = self._robot._device
    self._config = ConfigDict()
    self._config.initial_offset = to_torch(gait_config.initial_offset,
                                           device=self._device)
    self._config.swing_ratio = to_torch(gait_config.swing_ratio,
                                        device=self._device)
    self._config.stepping_frequency = gait_config.stepping_frequency
    self.reset()

  def reset(self):
    self._current_phase = torch.stack([self._config.initial_offset] *
                                      self._num_envs,
                                      axis=0).to(self._device)
    self._stepping_frequency = torch.ones(
        self._num_envs, device=self._device) * self._config.stepping_frequency
    self._swing_cutoff = torch.ones(
        (self._num_envs, 4),
        device=self._device) * 2 * torch.pi * (1 - self._config.swing_ratio)
    self._prev_frame_robot_time = self._robot.time_since_reset
    self._first_stance_seen = torch.zeros((self._num_envs, 4),
                                          dtype=torch.bool,
                                          device=self._device)

  def reset_idx(self, env_ids):
    self._current_phase[env_ids] = self._config.initial_offset
    self._stepping_frequency[env_ids] = self._config.stepping_frequency
    self._swing_cutoff[env_ids] = 2 * torch.pi * (1 - self._config.swing_ratio)
    self._prev_frame_robot_time[env_ids] = self._robot.time_since_reset[
        env_ids]
    self._first_stance_seen[env_ids] = 0

  def update(self):
    current_robot_time = self._robot.time_since_reset
    delta_t = current_robot_time - self._prev_frame_robot_time
    self._prev_frame_robot_time = current_robot_time
    self._current_phase += 2 * torch.pi * self._stepping_frequency[:,
                                                                   None] * delta_t[:,
                                                                                   None]

  @property
  def desired_contact_state(self):
    modulated_phase = torch.remainder(self._current_phase + 2 * torch.pi,
                                      2 * torch.pi)
    raw_contact = torch.where(modulated_phase > self._swing_cutoff, False,
                              True)
    # return raw_contact
    # print(f"Raw constact: {raw_contact}")
    self._first_stance_seen = torch.logical_or(self._first_stance_seen,
                                               raw_contact)
    return torch.where(self._first_stance_seen, raw_contact,
                       torch.ones_like(raw_contact))

  @property
  def desired_contact_state_state_estimation(self):
    """Only use the middle 50% of contact phase for state estimation."""
    modulated_phase = torch.remainder(self._current_phase + 2 * torch.pi,
                                      2 * torch.pi)
    raw_contact = torch.logical_and(
        modulated_phase > 0.25 * self._swing_cutoff,
        modulated_phase < 0.75 * self._swing_cutoff)
    return torch.where(self._first_stance_seen, raw_contact,
                       torch.ones_like(raw_contact))

  @property
  def normalized_phase(self):
    """Returns the leg's progress in the current state (swing or stance)."""
    modulated_phase = torch.remainder(self._current_phase + 2 * torch.pi,
                                      2 * torch.pi)
    return torch.where(modulated_phase < self._swing_cutoff,
                       modulated_phase / self._swing_cutoff,
                       (modulated_phase - self._swing_cutoff) /
                       (2 * torch.pi - self._swing_cutoff))

  @property
  def stance_duration(self):
    return (self._swing_cutoff) / (2 * torch.pi *
                                   self._stepping_frequency[:, None])

  @property
  def true_phase(self):
    return self._current_phase[:, 0] - self._config.initial_offset[0]

  @property
  def phase_obs(self):
    return self._current_phase - self._config.initial_offset

  @property
  def cycle_progress(self):
    true_phase = torch.remainder(self.true_phase + 2 * torch.pi, 2 * torch.pi)
    return true_phase / (2 * torch.pi)

  @property
  def stepping_frequency(self):
    return self._stepping_frequency

  @stepping_frequency.setter
  def stepping_frequency(self, new_frequency: torch.Tensor):
    self._stepping_frequency = new_frequency
