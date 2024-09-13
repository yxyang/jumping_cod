"""Behavior cloning with ground-truth state-actions."""
from ml_collections import ConfigDict

from src.agents.imitation_learning.policies import mlp_policy, mlp_policy_recurrent
from src.agents.ppo.configs import bound_together as bound_together_ppo
from src.envs.configs import bound_together
from src.envs import jump_env
from src.envs.terrain import GenerationMethod  # pylint: disable=unused-import


def get_camera_config():
  config = ConfigDict()

  config.horizontal_fov_deg = 150
  config.vertical_fov_deg = 150
  config.horizontal_res = 100
  config.vertical_res = 100
  config.position_in_base_frame = [0.3, 0., -0.03]
  config.orientation_rpy_in_base_frame = [0., 0.8, 0.]

  # config.horizontal_fov_deg = 87
  # config.vertical_fov_deg = 58
  # config.horizontal_res = 87
  # config.vertical_res = 58
  # config.position_in_base_frame = [0.3, 0., 0.]
  # config.orientation_rpy_in_base_frame = [0., 0., 0.]
  return config


def get_config():
  config = ConfigDict()

  env_config = bound_together.get_config()
  with env_config.unlocked():
    env_config.terrain.num_rows = 10
    env_config.terrain.num_cols = 10
    env_config.terrain.generation_method = GenerationMethod.UNIFORM_RANDOM
    env_config.terrain.curriculum = False
    env_config.terrain.random_terrain_type_within_level = True
    env_config.terrain.terrain_proportions = dict(
        slope_smooth=0.,
        slope_rough=0.,
        stair=1.,
        obstacles=0.,
        stepping_stones=0.,
        gap=0.,
        pit=1.,
    )
    env_config.camera_config = get_camera_config()

  config.env_config = env_config
  config.env_class = jump_env.JumpEnv

  config.teacher_config = bound_together_ppo.get_config()
  config.teacher_ckpt = 'logs/20230918/1_rnn_teacher/2023_09_18_14_43_34/model_2450.pt'
  # config.teacher_ckpt = 'logs/20230823/1_bound_sagittal_better_curriculum/2023_08_23_12_05_41/model_8000.pt'
  # with config.teacher_config.unlocked():
  #   config.teacher_config.training.runner.policy_class_name = 'ActorCritic'

  config.num_init_steps = 5000
  config.num_dagger_steps = 5000
  config.num_envs = 10

  config.batch_size = 5 # 256
  config.num_steps = 60 * 50

  # config.student_policy_class = mlp_policy_recurrent.MLPPolicyRecurrent
  config.student_policy_class = mlp_policy.MLPPolicy
  config.num_iters = 10

  return config
