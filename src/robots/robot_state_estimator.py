"""Simple state estimator for Go1 robot."""
import numpy as np
from filterpy.kalman import KalmanFilter

from src.utils.moving_window_filter import MovingWindowFilter

_DEFAULT_WINDOW_SIZE = 1
_ANGULAR_VELOCITY_FILTER_WINDOW_SIZE = 1


def convert_to_skew_symmetric(x: np.ndarray) -> np.ndarray:
  return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


class RobotStateEstimator:
  """Estimates base velocity of A1 robot.
  The velocity estimator consists of a state estimator for CoM velocity.
  Two sources of information are used:
  The integrated reading of accelerometer and the velocity estimation from
  contact legs. The readings are fused together using a Kalman Filter.
  """
  def __init__(self,
               robot,
               accelerometer_variance: np.ndarray = np.array(
                   [1.42072319e-05, 1.57958752e-05, 8.75317619e-05, 2e-5]),
               sensor_variance: np.ndarray = np.array(
                   [0.33705298, 0.14858707, 0.68439632, 0.68]) * 0.03,
               initial_variance: float = 0.1,
               use_external_contact_estimator: bool = False):
    """Initiates the velocity/height estimator.
    See filterpy documentation in the link below for more details.
    https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
    Args:
      robot: the robot class for velocity estimation.
      accelerometer_variance: noise estimation for accelerometer reading.
      sensor_variance: noise estimation for motor velocity reading.
      initial_covariance: covariance estimation of initial state.
    """
    self.robot = robot
    self._use_external_contact_estimator = use_external_contact_estimator
    self._foot_contact = np.ones(4)

    self.filter = KalmanFilter(dim_x=4, dim_z=4, dim_u=4)
    self.filter.x = np.array([0., 0., 0., 0.26])
    self._initial_variance = initial_variance
    self._accelerometer_variance = accelerometer_variance
    self._sensor_variance = sensor_variance
    self.filter.P = np.eye(4) * self._initial_variance  # State covariance
    self.filter.Q = np.eye(4) * accelerometer_variance
    self.filter.R = np.eye(4) * sensor_variance

    self.filter.H = np.eye(4)  # measurement function (y=H*x)
    self.filter.F = np.eye(4)  # state transition matrix
    self.filter.B = np.eye(4)

    self.ma_filter = MovingWindowFilter(window_size=_DEFAULT_WINDOW_SIZE)
    self._angular_velocity_filter = MovingWindowFilter(
        window_size=_ANGULAR_VELOCITY_FILTER_WINDOW_SIZE)
    self._angular_velocity = np.zeros(3)
    self._estimated_velocity = np.zeros(3)
    self._estimated_position = np.array([0., 0., self.robot.mpc_body_height])
    self.reset()

  def reset(self):
    self.filter.x = np.array([0., 0., 0., 0.26])
    self.filter.P = np.eye(4) * self._initial_variance
    self._last_timestamp = 0
    self._estimated_velocity = self.filter.x.copy()[:3]

  def _compute_delta_time(self, robot_state):
    del robot_state  # unused
    if self._last_timestamp == 0.:
      # First timestamp received, return an estimated delta_time.
      delta_time_s = self.robot.control_timestep
    else:
      delta_time_s = self.robot.raw_state.tick / 1000 - self._last_timestamp
    self._last_timestamp = self.robot.raw_state.tick / 1000
    return delta_time_s

  def _get_velocity_and_height_observation(self):
    foot_positions = self.robot.foot_positions_in_base_frame_numpy
    rot_mat = self.robot.base_rot_mat_numpy
    ang_vel_cross = convert_to_skew_symmetric(self._angular_velocity)
    observed_velocities, observed_heights = [], []
    if self._use_external_contact_estimator:
      foot_contact = self._foot_contact.copy()
    else:
      foot_contact = self.robot.foot_contacts_numpy
    for leg_id in range(4):
      if foot_contact[leg_id]:
        jacobian = self.robot.compute_foot_jacobian(leg_id)
        # Only pick the jacobian related to joint motors
        joint_velocities = self.robot.motor_velocities_numpy[leg_id *
                                                             3:(leg_id + 1) *
                                                             3]
        leg_velocity_in_base_frame = jacobian.dot(joint_velocities)[:3]
        observed_velocities.append(
            -rot_mat.dot(leg_velocity_in_base_frame +
                         ang_vel_cross.dot(foot_positions[leg_id])))
        observed_heights.append(-rot_mat.dot(foot_positions[leg_id])[2] + 0.02)

    return observed_velocities, observed_heights

  def update_foot_contact(self, foot_contact):
    self._foot_contact = foot_contact.cpu().numpy().reshape(4)

  def update(self, robot_state):
    """Propagate current state estimate with new accelerometer reading."""
    delta_time_s = self._compute_delta_time(robot_state)
    sensor_acc = np.array(robot_state.imu.accelerometer)
    rot_mat = self.robot.base_rot_mat_numpy
    calibrated_acc = np.zeros(4)
    calibrated_acc[:3] = rot_mat.dot(sensor_acc) + np.array([0., 0., -9.8])
    calibrated_acc[3] = self._estimated_velocity[2]
    self.filter.predict(u=calibrated_acc * delta_time_s)

    (observed_velocities,
     observed_heights) = self._get_velocity_and_height_observation()

    if observed_velocities:
      observed_velocities = np.mean(observed_velocities, axis=0)
      observed_heights = np.mean(observed_heights)
      self.filter.update(
          np.concatenate((observed_velocities, [observed_heights])))

    self._estimated_velocity = self.ma_filter.calculate_average(
        self.filter.x.copy()[:3])
    self._angular_velocity = self._angular_velocity_filter.calculate_average(
        np.array(robot_state.imu.gyroscope))

    self._estimated_position += delta_time_s * self._estimated_velocity
    self._estimated_position[2] = self.filter.x.copy()[3]

  @property
  def estimated_velocity(self):
    return self._estimated_velocity.copy()

  @property
  def estimated_position(self):
    return self._estimated_position.copy()

  @property
  def angular_velocity(self):
    return self._angular_velocity

  @property
  def use_external_contact_estimator(self):
    return self._use_external_contact_estimator

  @use_external_contact_estimator.setter
  def use_external_contact_estimator(self, use_external_contact_estimator):
    self._use_external_contact_estimator = use_external_contact_estimator
