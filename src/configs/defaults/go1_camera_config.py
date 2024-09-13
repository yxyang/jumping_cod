"""Default config for asset options."""
from ml_collections import ConfigDict


def get_config():
  config = ConfigDict()

  config.horizontal_fov_deg = 70.21
  config.vertical_fov_deg = 59.18
  config.horizontal_res = 60
  config.vertical_res = 48
  config.position_in_base_frame = [0.245 + 0.027, 0.0075, 0.072 + 0.02]
  config.orientation_rpy_in_base_frame = [0., 0.52, 0.]

  # config.horizontal_fov_deg = 87
  # config.vertical_fov_deg = 58
  # config.horizontal_res = 87
  # config.vertical_res = 58
  # config.position_in_base_frame = [0.3, 0., 0.]
  # config.orientation_rpy_in_base_frame = [0., 0., 0.]
  return config
