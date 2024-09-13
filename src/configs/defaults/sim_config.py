"""Default simulation config."""

from ml_collections import ConfigDict


def get_config(use_gpu=True,
               show_gui=True,
               sim_dt=0.002,
               use_penetrating_contact=True,
               use_real_robot=False):
  config = ConfigDict()

  if not use_real_robot:
    from isaacgym import gymapi
    config.physics_engine = gymapi.SIM_PHYSX
    sim_params = gymapi.SimParams()
    sim_params.use_gpu_pipeline = use_gpu
    sim_params.dt = sim_dt
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UpAxis(gymapi.UP_AXIS_Z)
    sim_params.gravity = gymapi.Vec3(0., 0., -9.81)
    sim_params.physx.use_gpu = use_gpu
    sim_params.physx.num_subscenes = 0  #default_args.subscenes
    sim_params.physx.num_threads = 10
    sim_params.physx.solver_type = 1  # 0: pgs, 1: tgs
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    if use_penetrating_contact:
      sim_params.physx.contact_offset = 0.
      sim_params.physx.rest_offset = -0.01
    else:
      sim_params.physx.contact_offset = 0.01
      sim_params.physx.rest_offset = 0.
    sim_params.physx.bounce_threshold_velocity = 0.5  #0.5 [m/s]
    sim_params.physx.max_depenetration_velocity = 1.0
    sim_params.physx.max_gpu_contact_pairs = 2**23  #2**24 needed for 8000+ envs
    sim_params.physx.default_buffer_size_multiplier = 5
    sim_params.physx.contact_collection = gymapi.ContactCollection(
        2)  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    config.sim_params = sim_params

  config.sim_device = 'cuda' if use_gpu else 'cpu'
  config.show_gui = show_gui
  config.action_repeat = 1
  config.dt = sim_dt
  return config
