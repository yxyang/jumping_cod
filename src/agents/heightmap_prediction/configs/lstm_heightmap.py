"""Behavior cloning with ground-truth state-actions."""
import os

from ml_collections import ConfigDict
import yaml

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
  config.teacher_ckpt = (
      'logs/20240811/1_ablation_ours/2024_08_13_17_29_11/model_7950.pt'
  )

  config_path = os.path.join(os.path.dirname(config.teacher_ckpt),
                             "config.yaml")
  with open(config_path, "r", encoding="utf-8") as f:
    rl_config = yaml.load(f, Loader=yaml.Loader)
    env_config = rl_config.environment

  with env_config.unlocked():
    # env_config.terrain.num_rows = 10
    # env_config.terrain.num_cols = 10
    env_config.terrain.generation_method = GenerationMethod.UNIFORM_RANDOM
    env_config.terrain.curriculum = False
    env_config.terrain.random_terrain_type_within_level = True
    env_config.use_full_qp = False
    # env_config.terrain.terrain_proportions = dict(
    #     slope_smooth=0.,
    #     slope_rough=0.,
    #     stair=0.,
    #     obstacles=0.,
    #     stepping_stones=0.,
    #     gap=0.5,
    #     pit=0.5,
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
