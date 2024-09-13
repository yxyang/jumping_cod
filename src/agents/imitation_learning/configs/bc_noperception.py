"""Behavior cloning with ground-truth state-actions."""
from ml_collections import ConfigDict

from src.agents.imitation_learning.policies import proprioceptive_policy
from src.agents.ppo.configs import bound_together as bound_together_ppo
from src.envs.configs import bound_together
from src.envs import jump_env
from src.envs.terrain import GenerationMethod  # pylint: disable=unused-import


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

  config.env_config = env_config
  config.env_class = jump_env.JumpEnv

  config.teacher_ckpt = 'logs/20230823/1_bound_sagittal_better_curriculum/2023_08_23_12_05_41/model_8000.pt'
  config.teacher_config = bound_together_ppo.get_config()
  config.num_init_steps = 5000
  config.num_dagger_steps = 5000
  config.num_envs = 10

  config.batch_size = 256
  config.num_steps = 20 * 50

  config.student_policy_class = proprioceptive_policy.ProprioceptivePolicy
  config.num_iters = 10

  return config
