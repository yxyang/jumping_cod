"""Interface for reading commands from Logitech F710 Gamepad."""
from absl import app
from absl import flags
from absl import logging

import enum
import itertools
import threading
import time

import evdev
from evdev import ecodes, ff
import numpy as np

FLAGS = flags.FLAGS
MAX_ABS_VAL = 128


class ControllerMode(enum.Enum):
  WALK = 1
  BOUND = 2
  PRONK = 3
  PRONK_TURN = 4


MODE_TO_WORLD = {
    ControllerMode.WALK: "Walk",
    ControllerMode.BOUND: "Bound",
    ControllerMode.PRONK: "Pronk",
    ControllerMode.PRONK_TURN: "Pronk Turn"
}

ALLOWED_MODES = [
    ControllerMode.WALK, ControllerMode.BOUND, ControllerMode.PRONK
]


def _interpolate(raw_reading, max_raw_reading, new_scale):
  return raw_reading / max_raw_reading * new_scale


def find_device(keyword="ontroller"):
  devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
  for device in devices:
    if keyword in device.name:
      return device

  return None


class Gamepad:
  """Interface for reading commands from Logitech F710 Gamepad.

  The control works as following:
  1) Press LB+RB at any time for emergency stop
  2) Use the left joystick for forward/backward/left/right walking.
  3) Use the right joystick for rotation around the z-axis.
  """
  def __init__(self,
               vel_scale_x: float = .5,
               vel_scale_y: float = .5,
               vel_scale_rot: float = 1.,
               max_acc: float = 2):
    """Initialize the gamepad controller.
    Args:
      vel_scale_x: maximum absolute x-velocity command.
      vel_scale_y: maximum absolute y-velocity command.
      vel_scale_rot: maximum absolute yaw-dot command.
    """
    self.gamepad = find_device()
    if self.gamepad is None:
      raise RuntimeError("Cannot find a working gamepad.")

    self._vel_scale_x = vel_scale_x
    self._vel_scale_y = vel_scale_y
    self._vel_scale_rot = vel_scale_rot
    self._lb_pressed = False
    self._rb_pressed = False
    self._lj_pressed = False
    self._walk_height = 0.
    self._foot_height = 0.

    self._mode_generator = itertools.cycle(ALLOWED_MODES)
    self._mode = next(self._mode_generator)

    # Controller states
    self.vx_raw, self.vy_raw, self.wz_raw = 0., 0., 0.
    self.vx, self.vy, self.wz = 0., 0., 0.
    self._max_acc = max_acc
    self._estop_flagged = False
    self.is_running = True
    self.last_timestamp = time.time()

    self.listen_thread = threading.Thread(target=self.listen)
    self.listen_thread.daemon = True
    self.listen_thread.start()

    print("To confirm that you are using the correct gamepad, press down the "
          "LEFT joystick to continue...")

    rumble_start_time = time.time()
    while time.time() - rumble_start_time < 10:
      rumble = ff.Rumble(strong_magnitude=0x0000, weak_magnitude=0x7fff)
      effect_type = ff.EffectType(ff_rumble_effect=rumble)
      effect = ff.Effect(ecodes.FF_RUMBLE, -1, 0, ff.Trigger(0, 0),
                         ff.Replay(200, 0), effect_type)
      effect_id = self.gamepad.upload_effect(effect)
      repeat_count = 1
      self.gamepad.write(ecodes.EV_FF, effect_id, repeat_count)
      start_time = time.time()
      while time.time() - start_time < 1:
        if self._lj_pressed:
          self.gamepad.erase_effect(effect_id)
          return
        time.sleep(0.01)
      self.gamepad.erase_effect(effect_id)

    raise RuntimeError("Gamepad response not detected after 10 seconds, "
                       "terminating...")

  def listen(self):
    while self.is_running:
      for event in self.gamepad.read_loop():
        if event.type == ecodes.EV_KEY:
          if event.code == ecodes.BTN_TL:
            if event.value == 1:
              self.on_L1_press()
            else:
              self.on_L1_release()
          elif event.code == ecodes.BTN_TR:
            if event.value == 1:
              self.on_R1_press()
            else:
              self.on_R1_release()
          elif event.code == ecodes.BTN_THUMBL:
            if event.value == 1:
              self.on_L3_press()
            else:
              self.on_L3_release()
          elif event.code == ecodes.BTN_NORTH:
            if event.value == 1:
              self.on_triangle_press()
            else:
              self.on_triangle_release()
          elif event.code == ecodes.BTN_WEST:
            if event.value == 1:
              self.on_square_press()
            else:
              self.on_square_release()
          elif event.code == ecodes.BTN_SOUTH:
            if event.value == 1:
              self.on_circle_press()
            else:
              self.on_circle_release()
        elif event.type == ecodes.EV_ABS:
          if event.code == ecodes.ABS_Y:
            self.on_ABS_Y(event.value)
          elif event.code == ecodes.ABS_X:
            self.on_ABS_X(event.value)
          elif event.code == ecodes.ABS_RY:
            self.on_ABS_RY(event.value)
          elif event.code == ecodes.ABS_RX:
            self.on_ABS_RX(event.value)

  def _check_estop(self):
    if self._lb_pressed and self._rb_pressed:
      if not self._estop_flagged:
        logging.info("EStop Flagged, press LEFT joystick to release.")
      self._estop_flagged = True
      self.vx_raw, self.vy_raw, self.wz_raw = 0., 0., 0.

  def on_L1_press(self):
    self._lb_pressed = True
    self._check_estop()

  def on_L1_release(self):
    self._lb_pressed = False

  def on_ABS_Y(self, value):
    value -= 128
    self.vx_raw = _interpolate(-value, MAX_ABS_VAL, self._vel_scale_x)

  def on_ABS_X(self, value):
    value -= 128
    self.wz_raw = _interpolate(-value, MAX_ABS_VAL, self._vel_scale_rot)

  def on_ABS_RX(self, value):
    value -= 128
    self.vy_raw = _interpolate(-value, MAX_ABS_VAL, self._vel_scale_y)

  def on_ABS_RY(self, value):
    pass

  def on_L3_press(self):
    self._lj_pressed = True
    if self._estop_flagged and self._lj_pressed:
      self._estop_flagged = False
      logging.info("Estop Released.")

  def on_L3_release(self):
    self._lj_pressed = False

  def on_R1_press(self):
    self._rb_pressed = True
    self._check_estop()

  def on_R1_release(self):
    self._rb_pressed = False

  def on_triangle_press(self):
    self._mode = ControllerMode.WALK
    print("Mode: WALK")

  def on_triangle_release(self):
    pass

  def on_square_press(self):
    self._mode = ControllerMode.BOUND
    print("Mode: BOUND")

  def on_square_release(self):
    pass

  def on_circle_press(self):
    self._mode = ControllerMode.PRONK
    print("Mode: PRONK")

  def on_circle_release(self):
    pass

  @property
  def speed_command(self):
    delta_time = np.minimum(time.time() - self.last_timestamp, 1)
    max_delta_speed = self._max_acc * delta_time
    self.vx = np.clip(self.vx_raw, self.vx - max_delta_speed,
                      self.vx + max_delta_speed)
    self.vy = np.clip(self.vy_raw, self.vy - max_delta_speed,
                      self.vy + max_delta_speed)
    self.wz = np.clip(self.wz_raw, self.wz - max_delta_speed,
                      self.wz + max_delta_speed)

    self.last_timestamp = time.time()
    return (self.vx, self.vy, 0), self.wz

  @property
  def estop_flagged(self):
    return self._estop_flagged

  def flag_estop(self):
    if not self._estop_flagged:
      logging.info("Estop flagged by program.")
      self._estop_flagged = True

  def wait_for_estop_clearance(self):
    while self._estop_flagged:
      time.sleep(0.01)

  @property
  def mode_command(self):
    return self._mode

  @property
  def gait_command(self):
    return self._gait

  def terminate(self):
    self.stop = True


def main(_):
  gamepad = Gamepad()
  while True:
    lin_speed, ang_speed = gamepad.speed_command
    estop = gamepad.estop_flagged
    vx, vy, _ = lin_speed
    print("Vx: {}, Vy: {}, Wz: {}, Estop: {}".format(vx, vy, ang_speed, estop))
    time.sleep(0.1)

  gamepad.terminate()


if __name__ == "__main__":
  app.run(main)
