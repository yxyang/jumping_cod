"""Configuration for running PPO using the Pointmass Env"""
from ml_collections import ConfigDict

from src.envs.configs import trot_together_speed
from src.envs import jump_env_speed


def get_training_config():
  config = ConfigDict()
  config.seed = 1

  policy_config = ConfigDict()
  policy_config.init_noise_std = .5
  policy_config.actor_hidden_dims = [512, 256, 128]
  policy_config.critic_hidden_dims = [512, 256, 128]
  policy_config.activation = "elu"
  config.policy = policy_config

  alg_config = ConfigDict()
  alg_config.value_loss_coef = 1.0
  alg_config.use_clipped_value_loss = True
  alg_config.clip_param = 0.2
  alg_config.entropy_coef = 0.01
  alg_config.num_learning_epochs = 5
  alg_config.num_mini_batches = 4
  alg_config.learning_rate = 1e-3
  alg_config.schedule = "adaptive"
  alg_config.gamma = 0.99
  alg_config.lam = 0.95
  alg_config.desired_kl = 0.01
  alg_config.max_grad_norm = 1.
  config.algorithm = alg_config

  runner_config = ConfigDict()
  runner_config.policy_class_name = "ActorCritic"
  runner_config.algorithm_class_name = "PPO"
  runner_config.num_steps_per_env = 24
  runner_config.save_interval = 50
  runner_config.experiment_name = "go1_single_jump"
  runner_config.max_iterations = 8000
  # runner_config.resume = False
  # runner_config.load_run = -1
  # runner_config.checkpoint = -1
  # runner_config.resume_path = None
  config.runner = runner_config
  return config


def get_config():
  config = ConfigDict()
  config.training = get_training_config()
  config.env_class = jump_env_speed.JumpEnv
  config.environment = trot_together_speed.get_config()
  return config
