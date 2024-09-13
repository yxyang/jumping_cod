"""Config for Go1 speed tracking environment."""
from ml_collections import ConfigDict
import numpy as np
import torch

from src.envs.terrain import GenerationMethod


def get_terrain_config():
  config = ConfigDict()

  # config.type = 'plane'
  config.type = 'trimesh'
  config.terrain_length = 10
  config.terrain_width = 10
  config.border_size = 15
  config.num_rows = 10
  config.num_cols = 10
  config.horizontal_scale = 0.05
  config.vertical_scale = 0.005
  config.move_up_distance = 4.5
  config.move_down_distance = 2.5
  config.slope_threshold = 0.75
  config.generation_method = GenerationMethod.CURRICULUM
  config.max_init_level = 1
  config.terrain_proportions = dict(
      slope_smooth=0.,
      slope_rough=0.,
      stair=0.4,
      obstacles=0.,
      stepping_stones=0.4,
      gap=0.1,
      pit=0.1,
  )
  config.randomize_steps = False
  config.randomize_step_width = True
  # Curriculum setup
  config.curriculum = True
  config.restitution = 0.
  return config


def get_config():
  config = ConfigDict()

  gait_config = ConfigDict()
  gait_config.stepping_frequency = 1
  gait_config.initial_offset = np.array([0.05, 0.05, -0.4, -0.4],
                                        dtype=np.float32) * (2 * np.pi)
  gait_config.swing_ratio = np.array([0.6, 0.6, 0.6, 0.6], dtype=np.float32)
  config.gait = gait_config

  config.goal_lb = torch.tensor([0.999, -1e-7], dtype=torch.float)  # R, Theta
  config.goal_ub = torch.tensor([1., 1e-7], dtype=torch.float)

  # Observation:
  config.observe_heights = True
  config.measured_points_x = list(np.linspace(-0.4, 0.8, 30))
  config.measured_points_y = [0.]
  config.height_noise_lb = [-0.08, 0., -0.05]
  config.height_noise_ub = [0.08, 0., 0.05]

  # Action: step_freq, height, vx, vy, vz, roll, pitch, roll_rate, pitch_rate, yaw_rate
  config.include_gait_action = True
  config.include_foot_action = True
  config.mirror_foot_action = True

  # Vel
  config.action_lb = np.array(
      [0.5, -0.001, -3, -0.01, -3, -0.001, -0.001, -0.01, -6., -0.01] +
      [-0.2, -0.07, 0.] * 2)
  config.action_ub = np.array(
      [3.999, 0.001, 3, 0.01, 3, 0.001, 0.001, 0.01, 6., 0.01] +
      [0.2, 0.07, 0.4] * 2)  # x: -0.3 - 0.3, y: 0 - 0.6 for large pit jumping
  config.use_swing_foot_reference = True



  config.episode_length_s = 20.
  config.max_jumps = 10
  config.env_dt = 0.01
  config.motor_strength_ratios = np.array([1., 1., 1.] * 4)  #(0.7, 1)  # 0.7
  config.motor_torque_delay_steps = 5
  config.use_yaw_feedback = False

  config.com_perturbation_lb = np.array([0., 0., 0., 0., -4., 0.])
  config.com_perturbation_ub = np.array([0., 0., 0., 0., 4., 0.])
  config.com_perturbation_refresh_rate = 2
  config.randomize_com = False

  config.foot_friction = 1.  #0.7
  config.base_position_kp = np.array([0., 0., 0.])
  config.base_position_kd = np.array([10., 10., 10.])
  config.base_orientation_kp = np.array([50., 0., 0.])
  config.base_orientation_kd = np.array([1., 10., 10.])
  config.qp_foot_friction_coef = 0.6
  config.qp_weight_ddq = np.diag([1., 1., 10., 10., 10., 1.])
  config.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 1.5

  # Swing foot limit in body f
  config.swing_foot_limit = (-0.4, -0.1) # (-0.4, 0.1) for large pit jumping

  config.use_full_qp = False
  config.clip_grf_in_sim = True
  config.swing_foot_height = 0.
  config.swing_foot_landing_clearance = 0.
  config.terminate_on_body_contact = True
  config.terminate_on_limb_contact = False
  config.use_penetrating_contact = False

  config.terrain = get_terrain_config()

  config.rewards = [
      ('alive', 0.02),
      # ('upright', 0.02),
      ('contact_consistency', 0.008),
      # ('energy_consumption', 1e-6),
      ('foot_slipping', 0.032),
      ('foot_clearance', 0.008),
      ('out_of_bound_action', 0.01),
      ('knee_contact', 0.064),  # 0.064
      ('stepping_freq', 0.008),
      ('com_distance_to_goal_squared', 0.016),
      ('vertical_distance', 0.016),
      ('com_yaw', 0.016),
      # ('body_contact', 0.1),
      # ('friction_cone', 0.008)
      ('com_height', 0.01),
      ('qp_cost', 1e-4),
      # ('vel_command', 0.002)
      # ('foot_force', 0.002),
      # ('swing_foot_vel', 0.002),
      # ('forward_speed', 0.02)
      # ('swing_residual', 0.01)
      # ('acc_consistency', 1e-6)
  ]
  config.clip_negative_reward = False
  config.normalize_reward_by_phase = True

  config.terminal_rewards = [
      # ('com_distance_to_goal_squared', 10.),
  ]
  config.clip_negative_terminal_reward = False
  return config
