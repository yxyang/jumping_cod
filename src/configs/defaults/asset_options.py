"""Default config for asset options."""
from isaacgym import gymapi
from ml_collections import ConfigDict


def get_config():
  config = ConfigDict()
  asset_options = gymapi.AssetOptions()
  asset_options.default_dof_drive_mode = 3
  asset_options.collapse_fixed_joints = True
  asset_options.replace_cylinder_with_capsule = True
  asset_options.flip_visual_attachments = True
  asset_options.fix_base_link = False
  asset_options.density = 0.001
  asset_options.angular_damping = 0.
  asset_options.linear_damping = 0.
  asset_options.max_angular_velocity = 1000.
  asset_options.max_linear_velocity = 1000.
  asset_options.armature = 0.
  asset_options.thickness = 0.01
  asset_options.disable_gravity = False
  config.asset_options = asset_options
  config.self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
  return config
