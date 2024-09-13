"""Behavior cloning with ground-truth state-actions."""
from ml_collections import ConfigDict

from src.agents.imitation_learning.policies import depth_policy, depth_policy_recurrent
from src.agents.ppo.configs import bound_together as bound_together_ppo
from src.agents.ppo.configs import trot_together_speed as trot_together_ppo
from src.envs.terrain import GenerationMethod  # pylint: disable=unused-import


def get_camera_config():
  config = ConfigDict()

  config.horizontal_fov_deg = 21.43
  config.vertical_fov_deg = 59.18
  config.horizontal_res = 16
  config.vertical_res = 48
  config.position_in_base_frame = [0.245 + 0.027, 0.0075, 0.072 + 0.02]
  config.orientation_rpy_in_base_frame = [0., 0.52, 0.]
  return config


def get_config():
  config = ConfigDict()

  config.teacher_config = bound_together_ppo.get_config()
  config.teacher_ckpt = 'logs/20240811/1_ablation_ours/2024_08_11_16_46_23/model_8000.pt'

  env_config = config.teacher_config.environment
  with env_config.unlocked():
    # env_config.terrain.num_rows = 10
    # env_config.terrain.num_cols = 10
    env_config.terrain.generation_method = GenerationMethod.UNIFORM_RANDOM
    env_config.terrain.curriculum = False
    env_config.terrain.random_terrain_type_within_level = True
    # env_config.terrain.terrain_proportions = dict(
    #     slope_smooth=0.,
    #     slope_rough=0.,
    #     stair=1.,
    #     obstacles=0.,
    #     stepping_stones=0.,
    #     gap=0.,
    #     pit=0.,
    # )
    env_config.camera_dt = 0.02
    env_config.camera_config = get_camera_config()
    env_config.episode_length_s = 10

  config.env_config = env_config
  config.env_class = config.teacher_config.env_class

  config.num_init_steps = 5000
  config.num_dagger_steps = 5000
  config.num_envs = 10

  config.batch_size = 4
  config.num_steps = 30 * 50

  config.student_policy_class = depth_policy_recurrent.DepthPolicyRecurrent
  config.num_iters = 30

  return config
