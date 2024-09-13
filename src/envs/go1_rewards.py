"""Set of rewards for the Go1 robot."""
import torch


class Go1Rewards:
  """Set of rewards for Go1 robot."""
  def __init__(self, env):
    self._env = env
    self._robot = self._env.robot
    self._gait_generator = self._env.gait_generator
    self._num_envs = self._env.num_envs
    self._device = self._env.device

  def speed_tracking_reward(self):
    return -torch.sum(
        torch.square(self._env._speed_cmds -
                     self._robot.base_velocity_body_frame[:, :2]),
        dim=1)

  def forward_speed_reward(self):
    return self._robot.base_velocity_world_frame[:, 0]

  def upright_reward(self):
    return self._robot.projected_gravity[:, 2]

  def alive_reward(self):
    return torch.ones(self._num_envs, device=self._device)

  def height_reward(self):
    return -torch.square(self._robot.base_position[:, 2] - 0.26)

  def foot_slipping_reward(self):
    foot_slipping = torch.sum(
        self._gait_generator.desired_contact_state * torch.sum(torch.square(
            self._robot.foot_velocities_in_world_frame[:, :, :2]),
                                                               dim=2),
        dim=1) / 4
    foot_slipping = torch.clip(foot_slipping, 0, 1)
    # print(self._gait_generator.desired_contact_state)
    # print(self._robot.foot_velocities_in_world_frame)
    # print(foot_slipping)
    # input("Any Key...")
    return -foot_slipping

  def foot_clearance_reward(self, foot_height_thres=0.02):
    desired_contacts = self._gait_generator.desired_contact_state
    foot_height = self._robot.foot_height - 0.02  # Foot radius
    # print(f"Foot height: {foot_height}")
    foot_height = torch.clip(foot_height, 0,
                             foot_height_thres) / foot_height_thres
    foot_clearance = torch.sum(
        torch.logical_not(desired_contacts) * foot_height, dim=1) / 4

    return foot_clearance

  def foot_force_reward(self):
    """Swing leg should not have contact force."""
    foot_forces = torch.norm(self._robot.foot_contact_forces, dim=2)
    calf_forces = torch.norm(self._robot.calf_contact_forces, dim=2)
    thigh_forces = torch.norm(self._robot.thigh_contact_forces, dim=2)
    limb_forces = (foot_forces + calf_forces + thigh_forces).clip(max=10)
    foot_mask = torch.logical_not(self._gait_generator.desired_contact_state)

    return -torch.sum(limb_forces * foot_mask, dim=1) / 4

  def cost_of_transport_reward(self):
    motor_power = torch.abs(0.3 * self._robot.motor_torques**2 +
                            self._robot.motor_torques *
                            self._robot.motor_velocities)
    commanded_vel = torch.sqrt(torch.sum(self._env.command[:, :2]**2, dim=1))
    return -torch.sum(motor_power, dim=1) / commanded_vel

  def energy_consumption_reward(self):
    motor_power = torch.clip(
        0.3 * self._robot.motor_torques**2 +
        self._robot.motor_torques * self._robot.motor_velocities,
        min=0)
    return -torch.clip(torch.sum(motor_power, dim=1), min=0., max=2000.)

  def contact_consistency_reward(self):
    desired_contact = self._gait_generator.desired_contact_state
    actual_contact = torch.logical_or(self._robot.foot_contacts,
                                      self._robot.calf_contacts)
    actual_contact = torch.logical_or(actual_contact,
                                      self._robot.thigh_contacts)
    # print(f"Actual contact: {actual_contact}")
    return torch.sum(desired_contact == actual_contact, dim=1) / 4

  def distance_to_goal_reward(self):
    base_position = self._robot.base_position_world
    return -torch.sqrt(
        torch.sum(torch.square(base_position[:, :2] -
                               self._env.desired_landing_position[:, :2]),
                  dim=1))

  def com_distance_to_goal_squared_reward(self):
    base_position = self._robot.base_position_world
    # print("Base position: {}".format(base_position))
    # print("Desired landing: {}".format(self._env.desired_landing_position))
    # import pdb
    # pdb.set_trace()
    # return -torch.sum(torch.square(
    #     (base_position[:, :2] - self._env.desired_landing_position[:, :2])),
    #                   dim=1)
    return -torch.sum(torch.square(
        (base_position[:, :2] - self._env.desired_landing_position[:, :2]) /
        (self._env._jumping_distance[:, None])),
                      dim=1)

  def com_distance_to_goal_squared_absolute_reward(self):
    base_position = self._robot.base_position_world
    return -torch.sum(torch.square(
        (base_position[:, :2] - self._env.desired_landing_position[:, :2])),
                      dim=1)

  def vertical_distance_reward(self):
    base_position = self._robot.base_position_world
    return -torch.square(base_position[:, 2] -
                         self._env.desired_landing_position[:, 2])

  def swing_foot_vel_reward(self):
    foot_vel = torch.sum(self._robot.foot_velocities_in_base_frame**2, dim=2)
    contact_mask = torch.logical_not(
        self._env.gait_generator.desired_contact_state)
    return -torch.sum(foot_vel * contact_mask, dim=1) / (
        torch.sum(contact_mask, dim=1) + 0.001)

  def com_height_reward(self):
    # Helps robot jump higher over all
    base_height = self._robot.base_position_world[:, 2]
    if self._env._terrain is not None:
      terrain_height = self._env._terrain.get_terrain_height_at(
          self._robot.base_position_world[:, :2])
    else:
      terrain_height = torch.zeros_like(base_height)
    return (base_height - terrain_height).clip(max=0.6)

  def com_yaw_reward(self):
    # Helps robot jump higher over all
    yaw_diff = self._robot.base_orientation_rpy[:, 2] - self._env._desired_yaw
    yaw_diff = torch.remainder(yaw_diff + torch.pi, 2 * torch.pi) - torch.pi
    return -torch.square(yaw_diff)

  def heading_reward(self):
    # print(self._robot.base_orientation_rpy[:, 2])
    # input("Any Key...")
    return -self._robot.base_orientation_rpy[:, 2]**2

  def out_of_bound_action_reward(self):
    exceeded_action = torch.maximum(
        self._env._action_lb - self._env._last_action,
        self._env._last_action - self._env._action_ub)
    exceeded_action = torch.clip(exceeded_action, min=0.)
    normalized_excess = exceeded_action / (self._env._action_ub -
                                           self._env._action_lb)
    return -torch.sum(torch.square(normalized_excess), dim=1)

  def swing_residual_reward(self):
    return -torch.mean(torch.square(self._env._last_action[:, -6:]), axis=1)

  def knee_contact_reward(self):
    rew = -((torch.sum(torch.logical_or(self._env.robot.thigh_contacts,
                                        self._env.robot.calf_contacts),
                       dim=1)).float()) / 4
    return rew

  def body_contact_reward(self):
    return -self._robot.has_body_contact.float()

  def stepping_freq_reward(self):
    """Reward for jumping at low frequency."""
    return 1.5 - self._env.gait_generator.stepping_frequency.clip(min=1.5)

  def friction_cone_reward(self):
    return -self._env._num_clips / 4

  def qp_cost_reward(self):
    return -torch.sqrt(self._env._qp_cost)

  def roll_reward(self):
    return -self._robot.base_orientation_rpy[:, 0]

  def vel_command_reward(self):
    ang_vel_diff = torch.sqrt(
        torch.mean((self._env._torque_optimizer.desired_angular_velocity -
                   self._robot.base_angular_velocity_gravity_frame)**2,
                  dim=1))
    lin_vel_diff = torch.sqrt(
        torch.mean((self._env._torque_optimizer.desired_linear_velocity -
                   self._robot.base_velocity_gravity_frame)**2,
                  dim=1))
    return -(ang_vel_diff + lin_vel_diff)
