"""Policy outputs desired CoM speed for Go1 to track the desired speed."""
# pytype: disable=attribute-error
from absl import logging

import itertools
import time
from typing import Sequence, Tuple

import ml_collections
import numpy as np
import torch

from src.configs.defaults import sim_config
from src.controllers import phase_gait_generator
from src.controllers import qp_torque_optimizer
from src.controllers import raibert_swing_leg_controller
from src.envs import go1_rewards
from src.envs import terrain
from src.robots import go1
from src.robots import go1_robot
from src.robots import gamepad_reader
from src.robots.motors import MotorControlMode
from src.utils.torch_utils import to_torch


# @torch.jit.script
def torch_rand_float(lower, upper, shape: Sequence[int], device: str):
  return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def gravity_frame_to_world_frame(robot_yaw, gravity_frame_vec):
  cos_yaw = torch.cos(robot_yaw)
  sin_yaw = torch.sin(robot_yaw)
  world_frame_vec = torch.clone(gravity_frame_vec)
  world_frame_vec[:, 0] = (cos_yaw * gravity_frame_vec[:, 0] -
                           sin_yaw * gravity_frame_vec[:, 1])
  world_frame_vec[:, 1] = (sin_yaw * gravity_frame_vec[:, 0] +
                           cos_yaw * gravity_frame_vec[:, 1])
  return world_frame_vec


@torch.jit.script
def world_frame_to_gravity_frame(robot_yaw, world_frame_vec):
  cos_yaw = torch.cos(robot_yaw)
  sin_yaw = torch.sin(robot_yaw)
  gravity_frame_vec = torch.clone(world_frame_vec)
  gravity_frame_vec[:, 0] = (cos_yaw * world_frame_vec[:, 0] +
                             sin_yaw * world_frame_vec[:, 1])
  gravity_frame_vec[:, 1] = (sin_yaw * world_frame_vec[:, 0] -
                             cos_yaw * world_frame_vec[:, 1])
  return gravity_frame_vec


def create_sim(sim_conf):
  from isaacgym import gymapi, gymutil
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
  return gym, sim, viewer


class JumpEnv:
  def __init__(self,
               num_envs: int,
               config: ml_collections.ConfigDict(),
               device: str = "cuda",
               show_gui: bool = False,
               use_real_robot: bool = False):
    self._num_envs = num_envs
    self._device = device
    self._show_gui = show_gui
    self._config = config
    self._use_real_robot = use_real_robot
    self._cmd_schedule = config.get('jumping_distance_schedule', None)
    if self._cmd_schedule is not None:
      self._cmd_schedule = itertools.cycle(self._cmd_schedule)
    with self._config.unlocked():
      self._config.goal_lb = to_torch(self._config.goal_lb,
                                      device=self._device)
      self._config.goal_ub = to_torch(self._config.goal_ub,
                                      device=self._device)
      if self._config.get('observation_noise', None) is not None:
        self._config.observation_noise = to_torch(
            self._config.observation_noise, device=self._device)

    # Set up robot and controller
    use_gpu = ("cuda" in device)
    self._sim_conf = sim_config.get_config(
        use_gpu=use_gpu,
        show_gui=show_gui,
        use_penetrating_contact=self._config.get('use_penetrating_contact',
                                                 False),
        use_real_robot=self._use_real_robot)
    if not self._use_real_robot:
      from isaacgym import gymapi
      self._gymapi = gymapi
      self._gym, self._sim, self._viewer = create_sim(self._sim_conf)
      self._create_terrain()
    else:
      self._gym, self._sim, self._viewer = None, None, None
      self._terrain = None

    self._init_positions = self._compute_init_positions()

    if self._use_real_robot:
      robot_class = go1_robot.Go1Robot
      self._gamepad = gamepad_reader.Gamepad()
    else:
      robot_class = go1.Go1
    self._robot = robot_class(
        num_envs=self._num_envs,
        init_positions=self._init_positions,
        sim=self._sim,
        viewer=self._viewer,
        sim_config=self._sim_conf,
        motor_control_mode=MotorControlMode.HYBRID,
        terrain=self._terrain,
        motor_torque_delay_steps=self._config.get('motor_torque_delay_steps',
                                                  0),
        camera_config=self._config.get('camera_config', None),
        randomize_com=self._config.get('randomize_com', False))
    strength_ratios = self._config.get('motor_strength_ratios', 0.7)
    if isinstance(strength_ratios, Sequence) and len(strength_ratios) == 2:
      ratios = torch_rand_float(lower=to_torch([strength_ratios[0]],
                                               device=self._device),
                                upper=to_torch([strength_ratios[1]],
                                               device=self._device),
                                shape=(self._num_envs, 3),
                                device=self._device)
      # Use the same ratio for all ab/ad motors, all hip motors, all knee motors
      ratios = torch.concatenate((ratios, ratios, ratios, ratios), dim=1)
      self._robot.motor_group.strength_ratios = ratios
    else:
      self._robot.motor_group.strength_ratios = strength_ratios

    # Need to set frictions twice to make it work on GPU... 😂
    self._robot.set_foot_frictions(0.01)
    self._robot.set_foot_frictions(self._config.get('foot_friction', 1.))
    if not self._use_real_robot:
      # Populate robot state tensors so that we know robot orientation.
      self._robot.reset()
    self._gait_generator = phase_gait_generator.PhaseGaitGenerator(
        self._robot, self._config.gait)
    self._swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        self._robot,
        self._gait_generator,
        foot_height=self._config.get('swing_foot_height', 0.),
        foot_landing_clearance=self._config.get('swing_foot_landing_clearance',
                                                0.))

    self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
        self._robot,
        self._gait_generator,
        base_position_kp=self._config.get('base_position_kp',
                                          np.array([0., 0., 50.])),
        base_position_kd=self._config.get('base_position_kd',
                                          np.array([10., 10., 10.])),
        base_orientation_kp=self._config.get('base_orientation_kp',
                                             np.array([50., 50., 0.])),
        base_orientation_kd=self._config.get('base_orientation_kd',
                                             np.array([10., 10., 10.])),
        weight_ddq=self._config.get('qp_weight_ddq',
                                    np.diag([20.0, 20.0, 5.0, 1.0, 1.0, .2])),
        foot_friction_coef=self._config.get('qp_foot_friction_coef', 0.7),
        clip_grf=self._config.get('clip_grf_in_sim') or self._use_real_robot,
        body_inertia=self._config.get('qp_body_inertia',
                                      np.array([0.14, 0.35, 0.35]) * 1.5),
        use_full_qp=self._config.get('use_full_qp', False),
        swing_foot_limit=self._config.get('swing_foot_limit', None))

    self._steps_count = torch.zeros(self._num_envs, device=self._device)
    self._episode_length = self._config.episode_length_s / self._config.env_dt
    self._construct_observation_and_action_space()
    if self._config.get("observe_heights", False):
      self._height_points, self._num_height_points = self._compute_height_points(
      )
      self._heightmap_bias = torch_rand_float(
          torch.tensor(self._config.get('height_noise_lb',
                                        [-1e-7, -1e-7, -1e-7]),
                       device=self._device),
          torch.tensor(self._config.get('height_noise_ub', [1e-7, 1e-7, 1e-7]),
                       device=self._device), [self._num_envs, 3],
          device=self._device)
    self._obs_buf = None
    self._privileged_obs_buf = None
    self._desired_landing_position = torch.zeros((self._num_envs, 3),
                                                 device=self._device,
                                                 dtype=torch.float)
    self._jumping_distance = torch.zeros(self._num_envs, device=self._device)
    self._desired_yaw = torch.clone(self._robot.base_orientation_rpy[:, 2])
    self._cycle_count = torch.zeros(self._num_envs, device=self._device)

    self._curr_com_perturbation = torch.zeros(self._num_envs,
                                              6,
                                              device=self._device)

    self._com_perturbation_refresh_rate = self._config.get(
        "com_perturbation_refresh_rate", 2.)
    self._time_since_perturbation = torch.ones(
        self._num_envs,
        device=self._device) / self._com_perturbation_refresh_rate + 1
    self._com_perturbation_lb = to_torch(self._config.get(
        'com_perturbation_lb', [0., 0., 0., 0., 0., 0.]),
                                         device=self._device)
    self._com_perturbation_ub = to_torch(self._config.get(
        'com_perturbation_ub', [0., 0., 0., 0., 0., 0.]),
                                         device=self._device)
    self._camera_refresh_frequency = self._config.get(
        'camera_dt', 0.1) / self._config.get('env_dt', 0.01)
    self._resample_command(torch.arange(self._num_envs, device=self._device))

    self._rewards = go1_rewards.Go1Rewards(self)
    self._prepare_rewards()
    self._extras = dict()

    # Running a few steps with dummy commands to ensure JIT compilation
    if self._num_envs == 1 and self._use_real_robot:
      for state in range(16):
        desired_contact_state = torch.tensor([[(state & (1 << i)) != 0
                                               for i in range(4)]],
                                             dtype=torch.bool,
                                             device=self._device)
        for _ in range(3):
          self._gait_generator.update()
          self._swing_leg_controller.update()
          desired_foot_positions = self._swing_leg_controller.desired_foot_positions
          self._torque_optimizer.get_action(
              desired_contact_state,
              swing_foot_position=desired_foot_positions)

  def _create_terrain(self):
    """Creates terrains.

    Note that we set the friction coefficient to all 0 here. This is because
    Isaac seems to pick the larger friction out of a contact pair as the
    actual friction coefficient. We will set the corresponding friction coef
    in robot friction.
    """
    if self._config.terrain.type == 'plane':
      plane_params = self._gymapi.PlaneParams()
      plane_params.normal = self._gymapi.Vec3(0.0, 0.0, 1.0)
      plane_params.static_friction = 1.
      plane_params.dynamic_friction = 1.
      plane_params.restitution = self._config.terrain.restitution
      self._gym.add_ground(self._sim, plane_params)
      self._terrain = None
    elif self._config.terrain.type == 'heightfield':
      self._terrain = terrain.Terrain(self._config.terrain,
                                      device=self._device)
      hf_params = self._gymapi.HeightFieldParams()
      hf_params.column_scale = self._config.terrain.horizontal_scale
      hf_params.row_scale = self._config.terrain.horizontal_scale
      hf_params.vertical_scale = self._config.terrain.vertical_scale
      hf_params.nbRows = self._terrain.total_cols
      hf_params.nbColumns = self._terrain.total_rows
      hf_params.transform.p.x = -self._config.terrain.border_size
      hf_params.transform.p.y = -self._config.terrain.border_size
      hf_params.transform.p.z = 0.0
      hf_params.static_friction = 1.
      hf_params.dynamic_friction = 1.
      hf_params.restitution = self._config.terrain.restitution
      self._gym.add_heightfield(self._sim, self._terrain.height_samples,
                                hf_params)
    elif self._config.terrain.type == 'trimesh':
      self._terrain = terrain.Terrain(self._config.terrain,
                                      device=self._device)
      tm_params = self._gymapi.TriangleMeshParams()
      tm_params.nb_vertices = self._terrain.vertices.shape[0]
      tm_params.nb_triangles = self._terrain.triangles.shape[0]

      tm_params.transform.p.x = -self._config.terrain.border_size
      tm_params.transform.p.y = -self._config.terrain.border_size
      tm_params.transform.p.z = 0.0
      tm_params.static_friction = 0.99
      tm_params.dynamic_friction = 0.99
      tm_params.restitution = self._config.terrain.restitution
      self._gym.add_triangle_mesh(self._sim,
                                  self._terrain.vertices.flatten(order='C'),
                                  self._terrain.triangles.flatten(order='C'),
                                  tm_params)
    else:
      raise ValueError('Invalid terrain type: {}'.format(
          self._config.terrain.type))

  def _compute_init_positions(self):
    init_positions = torch.zeros((self._num_envs, 3), device=self._device)

    if self._use_real_robot or self._config.terrain.type == 'plane':
      num_cols = int(np.sqrt(self._num_envs))
      distance = 1.
      for idx in range(self._num_envs):
        init_positions[idx, 0] = idx // num_cols * distance
        init_positions[idx, 1] = idx % num_cols * distance
        init_positions[idx, 2] = 0.268
    else:
      num_cols = self._config.terrain.num_cols
      init_positions = np.zeros((self._num_envs, 3))
      self._terrain_types = torch.randint(0,
                                          self._config.terrain.num_cols,
                                          size=(self._num_envs, ),
                                          device=self._device)
      self._terrain_levels = torch.randint(self._config.terrain.get(
          'min_init_level', 0),
                                           self._config.terrain.max_init_level,
                                           size=(self._num_envs, ),
                                           device=self._device)
      init_positions = self._terrain.env_origins[self._terrain_levels,
                                                 self._terrain_types]
      init_positions[:, 2] += 0.268

      # Sobol sequence of pseudo-random numbers to minimize collisions
      self._sobol_engine = torch.quasirandom.SobolEngine(dimension=2,
                                                         scramble=True)
      sobol_points = self._sobol_engine.draw(self._num_envs)
      sobol_points = (sobol_points - 0.5) * 2.5
      init_positions[:, :2] += to_torch(sobol_points, device=self._device)
      # init_positions[:, 0] = 7.5

      # init_positions[:, 1] += np.random.uniform(-1.5, 1.5, size=self._num_envs)
      # init_positions[:, 0] += np.random.uniform(-1.5, 1.5, size=self._num_envs)

    return to_torch(init_positions, device=self._device)

  def _compute_height_points(self):
    x = to_torch(self._config.measured_points_x, device=self._device)
    y = to_torch(self._config.measured_points_y, device=self._device)
    grid_x, grid_y = torch.meshgrid(x, y)
    num_height_points = len(x) * len(y)
    points = torch.zeros((self._num_envs, num_height_points, 2),
                         device=self._device)
    points[:, :, 0] = grid_x.flatten()
    points[:, :, 1] = grid_y.flatten()
    return points, num_height_points

  def _construct_observation_and_action_space(self):
    robot_lb = to_torch(
        [0., -3.14, -3.14, -1, -1, -4., -4., -10., -3.14, -3.14, -3.14] +
        [-0.5, -0.5, -0.4] * 4 + [-4., -4.] * 4,
        device=self._device)
    robot_ub = to_torch(
        [0.6, 3.14, 3.14, 1, 1, 4., 4., 10., 3.14, 3.14, 3.14] +
        [0.5, 0.5, 0.] * 4 + [4., 4.] * 4,
        device=self._device)
    # robot_lb = to_torch([-3.14, -3.14, -3.14, -3.14, -3.14] +
    #                     [-0.8, -1.04, -2.7] * 4 + [-3.14, -3.14, -3.14] * 4,
    #                     device=self._device)
    # robot_ub = to_torch([3.14, 3.14, 3.14, 3.14, 3.14] +
    #                     [0.8, 3.18, -0.91] * 4 + [3.14, 3.14, 3.14] * 4,
    #                     device=self._device)

    task_lb = to_torch([-2., -2.] + [0.] * 8, device=self._device)
    task_ub = to_torch([2., 2] + [1.] * 8, device=self._device)
    self._observation_lb = torch.concatenate((task_lb, robot_lb))
    self._observation_ub = torch.concatenate((task_ub, robot_ub))
    if self._config.get("observe_heights", False):
      num_heightpoints = len(self._config.measured_points_x) * len(
          self._config.measured_points_y)
      self._observation_lb = torch.concatenate(
          (self._observation_lb,
           torch.zeros(num_heightpoints, device=self._device) - 3))
      self._observation_ub = torch.concatenate(
          (self._observation_ub,
           torch.zeros(num_heightpoints, device=self._device) + 3))
    self._action_lb = to_torch(self._config.action_lb, device=self._device)
    self._action_ub = to_torch(self._config.action_ub, device=self._device)

  def _prepare_rewards(self):
    self._reward_names, self._reward_fns, self._reward_scales = [], [], []
    self._episode_sums = dict()
    for name, scale in self._config.rewards:
      self._reward_names.append(name)
      self._reward_fns.append(getattr(self._rewards, name + '_reward'))
      self._reward_scales.append(scale)
      self._episode_sums[name] = torch.zeros(self._num_envs,
                                             device=self._device)

    self._terminal_reward_names, self._terminal_reward_fns, self._terminal_reward_scales = [], [], []
    for name, scale in self._config.terminal_rewards:
      self._terminal_reward_names.append(name)
      self._terminal_reward_fns.append(getattr(self._rewards,
                                               name + '_reward'))
      self._terminal_reward_scales.append(scale)
      self._episode_sums[name] = torch.zeros(self._num_envs,
                                             device=self._device)

  def reset(self) -> torch.Tensor:
    ans = self.reset_idx(torch.arange(self._num_envs, device=self._device))
    if self._config.get('camera_config', None) and not self._use_real_robot:
      self._robot.step_graphics()

    # Extra buffer refreshing after full reset.
    self._obs_buf = self._get_observations()
    self._privileged_obs_buf = self._get_privileged_observations()

    return ans

  def _split_action(self, action):
    gait_action = None
    if self._config.get('include_gait_action', False):
      gait_action = action[:, :1]
      action = action[:, 1:]

    foot_action = None
    if self._config.get('include_foot_action', False):
      if self._config.get('mirror_foot_action', False):
        foot_action = action[:, -6:].reshape((-1, 2, 3))
        foot_action = torch.stack([
            foot_action[:, 0],
            foot_action[:, 0],
            foot_action[:, 1],
            foot_action[:, 1],
        ],
                                  dim=1)
        foot_action[:, [1, 3], 1] *= -1  # Flip swing for left legs
        action = action[:, :-6]
      else:
        foot_action = action[:, -12:].reshape(
            (-1, 4, 3))  #+ self._robot.hip_offset
        action = action[:, :-12]

    com_action = torch.clone(action)
    return gait_action, com_action, foot_action

  def _reset_robot_position(self, env_ids):
    if self._config.terrain.get('curriculum', False):
      horizontal_distance = torch.norm(self._robot.base_position[env_ids, :2] -
                                       self._init_positions[env_ids, :2],
                                       dim=1)
      vertical_distance = self._robot.base_position[
          env_ids, 2] - self._init_positions[env_ids, 2]
      vertical_distance = vertical_distance.clip(min=0)
      distance_traveled = horizontal_distance + 1 * vertical_distance
      move_up = (distance_traveled > self._config.terrain.move_up_distance)
      move_down = (distance_traveled < self._config.terrain.move_down_distance)
      self._terrain_levels[env_ids] += (1 * move_up - 1 * move_down)
      # Minimum level is 0
      self._terrain_levels[env_ids] = torch.clip(
          self._terrain_levels[env_ids], 0, self._config.terrain.num_rows + 1)
      # Reaching max -> send to random level
      self._terrain_levels[env_ids] = torch.where(
          self._terrain_levels[env_ids] == self._config.terrain.num_rows,
          torch.randint_like(self._terrain_levels[env_ids],
                             self._config.terrain.num_rows),
          self._terrain_levels[env_ids])

      if self._config.terrain.get('random_terrain_type_within_level', False):
        self._terrain_types[env_ids] = torch.randint(
            0,
            self._config.terrain.num_cols,
            size=(env_ids.shape[0], ),
            device=self._device)
    else:
      self._terrain_levels[env_ids] = torch.randint_like(
          self._terrain_levels[env_ids], self._config.terrain.num_rows)

    self._init_positions[env_ids] = self._terrain.env_origins[
        self._terrain_levels[env_ids], self._terrain_types[env_ids]]

    sobol_points = self._sobol_engine.draw(len(env_ids))
    sobol_points = (sobol_points - 0.5) * 2.5  #* 0 + 0.97
    self._init_positions[env_ids, :2] += to_torch(sobol_points,
                                                  device=self._device)

    self._init_positions[:, 2] += 0.268
    self._robot.update_init_positions(env_ids, self._init_positions[env_ids])

  def reset_idx(self, env_ids) -> torch.Tensor:
    # Aggregate rewards
    self._extras["time_outs"] = self._episode_terminated()
    if env_ids.shape[0] > 0:
      self._extras["episode"] = {}
      if not self._use_real_robot:
        if self._config.terrain.type != "plane":
          self._reset_robot_position(env_ids)
        if self._config.terrain.curriculum:
          self._extras["episode"]["terrain_level"] = torch.mean(
              self._terrain_levels.float())

      self._extras["episode"]["cycle_count"] = torch.mean(
          self._gait_generator.true_phase[env_ids]) / (2 * torch.pi)

      self._obs_buf = self._get_observations()
      self._privileged_obs_buf = self._get_privileged_observations()

      for reward_name in self._episode_sums.keys():
        if reward_name in self._reward_names:
          if self._config.get('normalize_reward_by_phase', False):
            self._extras["episode"]["reward_{}".format(
                reward_name)] = torch.mean(
                    self._episode_sums[reward_name][env_ids] /
                    (self._gait_generator.true_phase[env_ids] /
                     (2 * torch.pi)))
          else:
            # Normalize by time
            self._extras["episode"]["reward_{}".format(
                reward_name)] = torch.mean(
                    self._episode_sums[reward_name][env_ids] /
                    (self._steps_count[env_ids] * (self._config.env_dt)))

        if reward_name in self._terminal_reward_names:
          self._extras["episode"]["reward_{}".format(
              reward_name)] = torch.mean(
                  self._episode_sums[reward_name][env_ids] /
                  (self._cycle_count[env_ids].clip(min=1)))

        self._episode_sums[reward_name][env_ids] = 0

      self._steps_count[env_ids] = 0
      self._cycle_count[env_ids] = 0
      self._robot.reset_idx(env_ids)
      self._swing_leg_controller.reset_idx(env_ids)
      self._gait_generator.reset_idx(env_ids)
      self._torque_optimizer.reset_idx(env_ids)
      self._resample_command(env_ids)

    return self._obs_buf, self._privileged_obs_buf

  def _refresh_com_perturbations(self):
    self._time_since_perturbation += self._robot.control_timestep
    env_ids_to_refresh = (self._time_since_perturbation *
                          self._com_perturbation_refresh_rate > 1).nonzero(
                              as_tuple=False).flatten()
    new_perturbations = torch_rand_float(self._com_perturbation_lb,
                                         self._com_perturbation_ub,
                                         [len(env_ids_to_refresh), 6],
                                         device=self._device)
    self._curr_com_perturbation[env_ids_to_refresh] = new_perturbations
    self._time_since_perturbation[env_ids_to_refresh] = 0

  def step(self, action: torch.Tensor, base_height_override=None):
    if base_height_override is not None:
      base_height = base_height_override
    else:
      base_height = -self._obs_buf[:, -20]

    self._last_action = torch.clone(action)
    action = torch.clip(action, self._action_lb, self._action_ub)
    # print(self._last_action[:, 2])
    sum_reward = torch.zeros(self._num_envs, device=self._device)
    dones = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    self._steps_count += 1
    logs = []

    zero = torch.zeros(self._num_envs, device=self._device)
    gait_action, com_action, foot_action = self._split_action(action)
    start_time = self._robot.time_since_reset[0]
    while self._robot.time_since_reset[0] - start_time < self._config.env_dt:
      self._gait_generator.update()
      self._swing_leg_controller.update(base_height)
      self._torque_optimizer.update()
      self._refresh_com_perturbations()
      if self._use_real_robot:
        self._robot.state_estimator.update_foot_contact(
            self._gait_generator.desired_contact_state_state_estimation)  # pytype: disable=attribute-error
        self._robot.update_desired_foot_contact(
            self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error

      if gait_action is not None:
        self._gait_generator.stepping_frequency = gait_action[:, 0]

      # CoM pose action
      self._torque_optimizer.desired_base_position = torch.stack(
          (self._robot.base_position[:, 0], self._robot.base_position[:, 1],
           com_action[:, 0].clip(self._robot.base_position[:, 2] - 0.001,
                                 self._robot.base_position[:, 2] + 0.001)),
          dim=1)
      self._torque_optimizer.desired_linear_velocity = torch.stack(
          (torch.clip(com_action[:, 1],
                      self._robot.base_velocity_gravity_frame[:, 0] - 1,
                      self._robot.base_velocity_gravity_frame[:, 0] + 1),
           torch.clip(com_action[:, 2],
                      self._robot.base_velocity_gravity_frame[:, 1] - 1,
                      self._robot.base_velocity_gravity_frame[:, 1] + 1),
           com_action[:, 3]),
          dim=1)
      # self._torque_optimizer.desired_linear_velocity = torch.clip(
      #     com_action[:, 1:4], self._robot.base_velocity_gravity_frame - 2,
      #     self._robot.base_velocity_gravity_frame + 2)
      self._torque_optimizer.desired_base_orientation_rpy = torch.stack(
          (torch.clip(com_action[:, 4],
                      self._robot.base_orientation_rpy[:, 0] - 0.2,
                      self._robot.base_orientation_rpy[:, 0] + 0.2),
           torch.clip(com_action[:, 5],
                      self._robot.base_orientation_rpy[:, 1] - 0.6,
                      self._robot.base_orientation_rpy[:, 1] + 0.6),
           self._robot.base_orientation_rpy[:, 2]),
          dim=1)

      self._torque_optimizer.desired_angular_velocity = torch.stack(
          (com_action[:, 6], com_action[:, 7],
           torch.clip(
               com_action[:, 8],
               self._robot.base_angular_velocity_gravity_frame[:, 2] - 1,
               self._robot.base_angular_velocity_gravity_frame[:, 2] + 1)),
          dim=1)
      # self._torque_optimizer.desired_angular_velocity = torch.clip(
      #     com_action[:,
      #                6:9], self._robot.base_angular_velocity_gravity_frame - 3,
      #     self._robot.base_angular_velocity_gravity_frame + 3)

      desired_foot_positions = self._swing_leg_controller.desired_foot_positions
      if foot_action is not None:
        # With Residual
        if self._config.get("use_swing_foot_reference", True):
          base_yaw = self._robot.base_orientation_rpy[:, 2]
          cos_yaw = torch.cos(base_yaw)[:, None]
          sin_yaw = torch.sin(base_yaw)[:, None]
          foot_action_world = torch.clone(foot_action)
          foot_action_world[:, :, 0] = (cos_yaw * foot_action[:, :, 0] -
                                        sin_yaw * foot_action[:, :, 1])
          foot_action_world[:, :, 1] = (sin_yaw * foot_action[:, :, 0] +
                                        cos_yaw * foot_action[:, :, 1])
          desired_foot_positions += foot_action_world
        else:
          # Without Residual, just gravity frame foot position
          desired_foot_positions = foot_action + torch.bmm(
              self._robot.base_rot_mat_zero_yaw,
              self._robot.hip_positions_in_body_frame.transpose(
                  1, 2)).transpose(1, 2)
          desired_foot_positions = torch.bmm(
              self._robot.base_rot_mat_yaw,
              desired_foot_positions.transpose(1, 2)).transpose(1, 2)
          # desired_foot_positions[:, :,
          #                        2] = torch.max(desired_foot_positions[:, :, 2],
          #                                       -base_height[:, None])

      motor_action, self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = self._torque_optimizer.get_action(
          self._gait_generator.desired_contact_state,
          swing_foot_position=desired_foot_positions)

      logs.append(
          dict(timestamp=self._robot.time_since_reset,
               base_position=torch.clone(self._robot.base_position),
               base_orientation_rpy=torch.clone(
                   self._robot.base_orientation_rpy),
               base_velocity=torch.clone(self._robot.base_velocity_body_frame),
               base_angular_velocity=torch.clone(
                   self._robot.base_angular_velocity_body_frame),
               motor_positions=torch.clone(self._robot.motor_positions),
               motor_velocities=torch.clone(self._robot.motor_velocities),
               motor_action=motor_action,
               motor_torques=self._robot.motor_torques,
               num_clips=self._num_clips,
               foot_contact_state=self._gait_generator.desired_contact_state,
               foot_contact_state_se=self._gait_generator.
               desired_contact_state_state_estimation,
               foot_contact_force=self._robot.foot_contact_forces,
               desired_swing_foot_position=desired_foot_positions,
               desired_acc_body_frame=self._desired_acc,
               solved_acc_body_frame=self._solved_acc,
               foot_positions_in_base_frame=self._robot.
               foot_positions_in_base_frame,
               env_action=action,
               env_obs=torch.clone(self._obs_buf)))
      # print(action[:, 2])
      if self._use_real_robot:
        logs[-1]["base_acc"] = np.array(
            self._robot.raw_state.imu.accelerometer)  # pytype: disable=attribute-error
        logs[-1]["tick"] = self._robot.raw_state.tick

      self._robot.step(motor_action,
                       perturbation_forces=self._curr_com_perturbation)

      self._obs_buf = self._get_observations()
      self._privileged_obs_buf = self.get_privileged_observations()
      rewards = self.get_reward()
      dones = torch.logical_or(dones, self._is_done())
      sum_reward += rewards * torch.logical_not(dones)

    if self._config.get('camera_config', None) and not self._use_real_robot:
      if (self._steps_count[0] % self._camera_refresh_frequency
          == 0) or dones[0]:
        self._robot.step_graphics()
        depth_imgs = []
        for i in range(self._num_envs):
          depth_imgs.append(self._robot.get_camera_image(i, mode='depth'))
        depth_imgs = torch.stack(depth_imgs, dim=0)
        logs[-1]["depth_image"] = (-depth_imgs / 3).clip(0, 1)
    # print(f"Time: {self._robot.time_since_reset}")
    # print(f"Gait: {gait_action}")
    # print(f"Foot: {foot_action}")
    # print(f"Phase: {self._obs_buf[:, 3]}")
    # print(f"Desired contact: {self._gait_generator.desired_contact_state}")
    # print(f"Desired Position: {self._torque_optimizer.desired_base_position}")
    # print(f"Current Position: {self._robot.base_position}")
    # print(
    #     f"Desired Velocity: {self._torque_optimizer.desired_linear_velocity}")
    # print(f"Current Velocity: {self._robot.base_velocity_world_frame}")
    # print(
    #     f"Desired RPY: {self._torque_optimizer.desired_base_orientation_rpy}")
    # print(f"Current RPY: {self._robot.base_orientation_rpy}")
    # print(
    #     f"Desired Angular Vel: {self._torque_optimizer.desired_angular_velocity}"
    # )
    # print(
    #     f"Current Angular vel: {self._robot.base_angular_velocity_body_frame}")
    # print(f"Desired Acc: {self._desired_acc}")
    # print(f"Solved Acc: {self._solved_acc}")
    # print(f"Num Clips: {self._num_clips}, QP Cost: {self._qp_cost}")
    # ans = input("Any Key...")
    # if ans in ["Y", "y"]:
    #   import pdb
    #   pdb.set_trace()
    self._extras["logs"] = logs
    # Resample commands
    new_cycle_count = (self._gait_generator.true_phase / (2 * torch.pi)).long()
    finished_cycle = new_cycle_count > self._cycle_count
    env_ids_to_resample = finished_cycle.nonzero(as_tuple=False).flatten()
    self._cycle_count = new_cycle_count

    is_terminal = torch.logical_or(finished_cycle, dones)
    if is_terminal.any():
      sum_reward += self.get_terminal_reward(is_terminal, dones)
      if self._use_real_robot:
        self._gamepad.terminate()
      # print(self.get_terminal_reward(is_terminal))
    # if dones.any():
    #   import pdb
    #   pdb.set_trace()
    self._resample_command(env_ids_to_resample)
    if not self._use_real_robot:
      self.reset_idx(dones.nonzero(as_tuple=False).flatten())

    if self._show_gui:
      self._robot.render()
    return self._obs_buf, self._privileged_obs_buf, sum_reward, dones, self._extras

  def _resample_command(self, env_ids):
    if env_ids.shape[0] == 0:
      return

    if self._num_envs == 1 and self._cmd_schedule is not None:
      cmd = torch.tensor([[next(self._cmd_schedule), 0.]], device=self._device)
    else:
      cmd = torch_rand_float(self._config.goal_lb,
                             self._config.goal_ub, [env_ids.shape[0], 2],
                             device=self._device)
    self._jumping_distance[env_ids], delta_yaw = cmd[:, 0], cmd[:, 1]
    if True or self._use_real_robot:
      self._desired_yaw[env_ids] = (
          self._robot.base_orientation_rpy[env_ids, 2] + delta_yaw).clip(
              min=-10000, max=10000)
    else:
      self._desired_yaw[env_ids] += delta_yaw

    self._desired_landing_position[env_ids, 0] = self._robot.base_position[
        env_ids, 0] + self._jumping_distance[env_ids] * torch.cos(
            self._desired_yaw[env_ids])
    self._desired_landing_position[env_ids, 1] = self._robot.base_position[
        env_ids, 1] + self._jumping_distance[env_ids] * torch.sin(
            self._desired_yaw[env_ids])

    self._desired_landing_position[env_ids, 2] = 0.268
    if self._terrain is not None:
      self._desired_landing_position[
          env_ids, 2] += self._terrain.get_terrain_height_at(
              self._desired_landing_position[env_ids, :2])

  def get_perception_base_states(self):
    base_vel_odo = torch.matmul(
        self._robot.base_rot_mat_zero_yaw,
        self._robot.base_velocity_body_frame[:, :, None])[:, :, 0]
    base_ang_vel_odo = torch.matmul(
        self._robot.base_rot_mat_zero_yaw,
        self._robot.base_angular_velocity_body_frame[:, :, None])[:, :, 0]
    return torch.concatenate(
        (
            self._robot.base_orientation_rpy[:, 0:1],  # Base roll
            self._robot.base_orientation_rpy[:, 1:2],  # Base Pitch
            base_vel_odo[:, 0:1] * 0,
            base_vel_odo[:, 1:2] * 0,
            base_vel_odo[:, 2:3] * 0,  # Base velocity (z)
            base_ang_vel_odo[:, 0:1] * 0,
            base_ang_vel_odo[:, 1:2] * 0,
            base_ang_vel_odo[:, 2:3] * 0,  # Base yaw rate
        ),
        dim=1)

  def get_proprioceptive_observation(self):
    distance_to_goal = self._desired_landing_position - self._robot.base_position_world

    distance_to_goal_local = world_frame_to_gravity_frame(
        self._robot.base_orientation_rpy[:, 2], distance_to_goal)
    # phase_obs = torch.concatenate((
    #     torch.cos(self._gait_generator.phase_obs),
    #     torch.sin(self._gait_generator.phase_obs),
    # ),
    #                         dim=1)
    phase_obs = torch.concatenate((
        self._gait_generator.desired_contact_state,
        self._gait_generator.normalized_phase,
    ),
                                  dim=1)

    base_vel_odo = torch.matmul(
        self._robot.base_rot_mat_zero_yaw,
        self._robot.base_velocity_body_frame[:, :, None])[:, :, 0]
    base_ang_vel_odo = torch.matmul(
        self._robot.base_rot_mat_zero_yaw,
        self._robot.base_angular_velocity_body_frame[:, :, None])[:, :, 0]
    robot_obs = torch.concatenate(
        (
            self._robot.base_position[:, 2:] * 0.,  # Base height
            self._robot.base_orientation_rpy[:, 0:1] * 0,  # Base roll
            self._robot.base_orientation_rpy[:, 1:2],  # Base Pitch
            torch.cos(self._robot.base_orientation_rpy[:, 2] -
                      self._desired_yaw)[:, None] * 0,
            torch.sin(self._robot.base_orientation_rpy[:, 2] -
                      self._desired_yaw)[:, None] * 0,
            base_vel_odo[:, 0:1],
            base_vel_odo[:, 1:2] * 0,
            base_vel_odo[:, 2:3],  # Base velocity (z)
            base_ang_vel_odo[:, 0:1] * 0,
            base_ang_vel_odo[:, 1:2] +
            torch.randn(self._num_envs, device=self._device)[:, None] * 0,
            base_ang_vel_odo[:, 2:3] * 0,  # Base yaw rate
            self._robot.foot_positions_in_gravity_frame.reshape(
                (self._num_envs, 12)),
            self._robot.foot_velocities_in_base_frame[:, :, [0, 2]].reshape(
                (self._num_envs, 8)) * 0.,
        ),
        dim=1)
    obs = torch.concatenate(
        (distance_to_goal_local[:, :2], phase_obs, robot_obs), dim=1)
    # if self._config.get("observation_noise",
    #                     None) is not None and (not self._use_real_robot):
    #   obs += torch.randn_like(obs) * self._config.observation_noise
    return obs

  def _get_observations(self):
    obs = self.get_proprioceptive_observation()
    if self._config.get("observe_heights", False):
      obs = torch.concatenate((obs, self.get_heights()), dim=1)
    return obs

  def get_heights(self, ground_truth: bool = False):
    if self._config.terrain.type == 'plane':
      return torch.zeros(
          (self._num_envs, self._num_height_points),
          device=self._device) - self._robot.base_position[:, 2, None]
    points_world = torch.zeros((self._num_envs, self._num_height_points, 2),
                               device=self._device)
    base_yaw = self._robot.base_orientation_rpy[:, 2]
    cos_yaw, sin_yaw = torch.cos(base_yaw)[:, None], torch.sin(base_yaw)[:,
                                                                         None]
    points_world[:, :, 0] = (cos_yaw * self._height_points[:, :, 0] -
                             sin_yaw * self._height_points[:, :, 1])
    points_world[:, :, 1] = (sin_yaw * self._height_points[:, :, 0] +
                             cos_yaw * self._height_points[:, :, 1])
    points_world += self._robot.base_position_world[:, None, :2]
    if not ground_truth:
      points_world += self._heightmap_bias[:, None, :2]

    heights = self._terrain.get_terrain_height_at(points_world.reshape(
        (-1, 2)))
    heights = heights.reshape((self._num_envs, self._num_height_points))
    heights -= self._robot.base_position_world[:, 2:]
    if not ground_truth:
      heights += self._heightmap_bias[:, None, 2]
    return heights

  def get_observations(self):
    return self._obs_buf

  def _get_privileged_observations(self):
    return None

  def get_privileged_observations(self):
    return self._privileged_obs_buf

  def get_reward(self):
    sum_reward = torch.zeros(self._num_envs, device=self._device)
    if not self._use_real_robot:
      for idx in range(len(self._reward_names)):
        reward_name = self._reward_names[idx]
        reward_fn = self._reward_fns[idx]
        reward_scale = self._reward_scales[idx]
        reward_item = reward_scale * reward_fn()
        if self._config.get('normalize_reward_by_phase', False):
          reward_item *= self._gait_generator.stepping_frequency
        self._episode_sums[reward_name] += reward_item
        sum_reward += reward_item

      if self._config.clip_negative_reward:
        sum_reward = torch.clip(sum_reward, min=0)
    return sum_reward

  def get_terminal_reward(self, is_terminal, dones):
    early_term = torch.logical_and(
        dones, torch.logical_not(self._episode_terminated()))
    coef = torch.where(early_term, self._gait_generator.cycle_progress,
                       torch.ones_like(early_term))

    sum_reward = torch.zeros(self._num_envs, device=self._device)
    for idx in range(len(self._terminal_reward_names)):
      reward_name = self._terminal_reward_names[idx]
      reward_fn = self._terminal_reward_fns[idx]
      reward_scale = self._terminal_reward_scales[idx]
      reward_item = reward_scale * reward_fn() * is_terminal * coef
      self._episode_sums[reward_name] += reward_item
      sum_reward += reward_item

    if self._config.clip_negative_terminal_reward:
      sum_reward = torch.clip(sum_reward, min=0)
    return sum_reward

  def _episode_terminated(self):
    timeout = (self._steps_count >= self._episode_length)
    cycles_finished = (self._gait_generator.true_phase /
                       (2 * torch.pi)) > self._config.get('max_jumps', 1)
    return torch.logical_or(timeout, cycles_finished)

  def _is_done(self):
    is_unsafe = torch.logical_or(self._robot.projected_gravity[:, 2] < 0.1,
                                 self._robot.base_position[:, 2] < 0.1)
    if self._config.get('terminate_on_body_contact', False):
      is_unsafe = torch.logical_or(is_unsafe, self._robot.has_body_contact)

    if self._config.get('terminate_on_limb_contact', False):
      limb_contact = torch.logical_or(self._robot.calf_contacts,
                                      self._robot.thigh_contacts)
      limb_contact = torch.sum(limb_contact, dim=1)
      is_unsafe = torch.logical_or(is_unsafe, limb_contact > 0)

    if self._use_real_robot:
      if self._gamepad.estop_flagged:
        is_unsafe = torch.logical_or(is_unsafe, torch.ones_like(is_unsafe))

    # print(self._robot.base_position[:, 2])
    # input("Any Key...")
    # if is_unsafe.any():
    #   import pdb
    #   pdb.set_trace()
    return torch.logical_or(self._episode_terminated(), is_unsafe)

  @property
  def device(self):
    return self._device

  @property
  def robot(self):
    return self._robot

  @property
  def gait_generator(self):
    return self._gait_generator

  @property
  def desired_landing_position(self):
    return self._desired_landing_position

  @property
  def action_space(self):
    return self._action_lb, self._action_ub

  @property
  def observation_space(self):
    return self._observation_lb, self._observation_ub

  @property
  def num_envs(self):
    return self._num_envs

  @property
  def num_obs(self):
    return self._observation_lb.shape[0]

  @property
  def num_privileged_obs(self):
    return None

  @property
  def num_actions(self):
    return self._action_lb.shape[0]

  @property
  def max_episode_length(self):
    return self._episode_length

  @property
  def episode_length_buf(self):
    return self._steps_count

  @episode_length_buf.setter
  def episode_length_buf(self, new_length: torch.Tensor):
    self._steps_count = to_torch(new_length, device=self._device)
    self._gait_generator._current_phase += 2 * torch.pi * (
        new_length / self.max_episode_length *
        self._config.get('max_jumps', 1) + 1)[:, None]
    self._cycle_count = (self._gait_generator.true_phase /
                         (2 * torch.pi)).long()

  @property
  def device(self):
    return self._device

  @property
  def terrain_levels(self):
    return self._terrain_levels

  @terrain_levels.setter
  def terrain_levels(self, new_levels: torch.Tensor):
    self._terrain_levels = new_levels
