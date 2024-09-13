"""Publishes Realsense images to ROS topics.

Camera Intrinsics Documentation:

=========================NEW for Linescan==============================
Realsense Raw Image
Resolution: 424x240
FoV: 90.18 x 59.18
cx, cy = 213.023, 119.512
fx, fy = 211.313, 211.313

After Crop (left, right by 172, top, bottom by 0):
Resolution: 80x240
FoV: 21.43 x 59.18
cx, cy = 40.023, 119.512
fx, fy = 211.313, 211.313

After resize (by 5x):
Resolution: 16x48
FoV: 8.01 x 59.18

cx, cy = 5.002, 23.902
fx, fy = 42.263, 42.263




=========================OLD for Imitation Learning====================
Realsense Raw Image
Resolution: 424x240
FoV: 90.18 x 59.18
cx, cy = 213.023, 119.512
fx, fy = 211.313, 211.313

After Crop (left, right by 62, top, bottom by 0):
Resolution: 300x240
FoV: 70.21 x 59.18
cx, cy = 151.023, 119.512
fx, fy = 211.313, 211.313

After resize (by 5x):
Resolution: 60x48
FoV: 70.21 x 59.18

cx, cy = 30.205, 23.902
fx, fy = 42.263, 42.263
"""
from absl import app
from absl import logging

from functools import partial

import cv2
from cv_bridge import CvBridge
import numpy as np
import pyrealsense2 as rs
import rospy
from sensor_msgs.msg import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F


@torch.no_grad()
def resize2d(img, size):
  return (F.adaptive_avg_pool2d(Variable(img), size)).data


def get_input_filter():
  """ Crop to match sim. """
  image_resolution = [48, 16]
  depth_range = [0., 3.]
  depth_range = (depth_range[0] * 1000, depth_range[1] * 1000)  # [m] -> [mm]
  crop_top, crop_bottom, crop_left, crop_right = 0, 0, 172, 172
  crop_far = 3.0 * 1000

  def input_filter(
      depth_image: torch.Tensor,
      crop_top: int,
      crop_bottom: int,
      crop_left: int,
      crop_right: int,
      crop_far: float,
      depth_min: int,
      depth_max: int,
      output_height: int,
      output_width: int,
  ):
    """ depth_image must have shape [1, 1, H, W] """
    depth_image = depth_image[:, :, crop_top:-crop_bottom - 1,
                              crop_left:-crop_right - 1, ]
    depth_image[depth_image > crop_far] = depth_max
    depth_image = torch.clip(
        depth_image,
        depth_min,
        depth_max,
    ) / (depth_max - depth_min)
    depth_image = resize2d(depth_image, (output_height, output_width))
    return depth_image

  # input_filter = torch.jit.script(input_filter)

  return partial(
      input_filter,
      crop_top=crop_top,
      crop_bottom=crop_bottom,
      crop_left=crop_left,
      crop_right=crop_right,
      crop_far=crop_far,
      depth_min=depth_range[0],
      depth_max=depth_range[1],
      output_height=image_resolution[0],
      output_width=image_resolution[1],
  )


def get_rs_filters():
  # build the sensor filter
  hole_filling_filter = rs.hole_filling_filter(2)
  spatial_filter = rs.spatial_filter()
  spatial_filter.set_option(rs.option.filter_magnitude, 5)
  spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
  spatial_filter.set_option(rs.option.filter_smooth_delta, 1)
  spatial_filter.set_option(rs.option.holes_fill, 4)
  temporal_filter = rs.temporal_filter()
  temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.75)
  temporal_filter.set_option(rs.option.filter_smooth_delta, 1)

  def filter_func(frame):
    frame = hole_filling_filter.process(frame)
    frame = spatial_filter.process(frame)
    frame = temporal_filter.process(frame)
    return frame

  return filter_func


def main(argv):
  del argv  # Unused.
  # Initialize ROS node
  rospy.init_node('realsense_publisher', anonymous=True)

  # Create publishers for color and depth images
  depth_gray_pub = rospy.Publisher('/camera/depth/image_grayscale',
                                   Image,
                                   queue_size=1)
  depth_input_pub = rospy.Publisher('/camera/depth/cnn_input',
                                    Image,
                                    queue_size=1,
                                    tcp_nodelay=True)
  # Configure RealSense camera
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 90)
  pipeline.start(config)
  logging.info("Pipeline started.")

  # CvBridge for converting images
  rs_filters = get_rs_filters()
  input_filter = get_input_filter()
  bridge = CvBridge()

  try:
    while not rospy.is_shutdown():
      # Wait for a coherent pair of frames: depth and color
      frames = pipeline.wait_for_frames()
      # color_frame = frames.get_color_frame()
      depth_frame = frames.get_depth_frame()
      if not depth_frame:  # or not color_frame:
        continue

      depth_frame = rs_filters(depth_frame)
      depth_image = np.asanyarray(depth_frame.get_data())
      depth_image = torch.from_numpy(depth_image.astype(
          np.float32)).unsqueeze(0).unsqueeze(0)
      depth_image = input_filter(depth_image)
      depth_image_numpy = depth_image.cpu().numpy()[0, 0]

      # Convert images to ROS Image message format
      normalized_depth = cv2.normalize(depth_image_numpy, None, 0, 255,
                                       cv2.NORM_MINMAX)
      depth_gray = normalized_depth.astype('uint8')
      depth_gray_message = bridge.cv2_to_imgmsg(depth_gray, encoding="mono8")

      depth_input_message = bridge.cv2_to_imgmsg(depth_image_numpy,
                                                 encoding="32FC1")

      # Publish the images
      depth_gray_pub.publish(depth_gray_message)
      depth_input_pub.publish(depth_input_message)

  finally:
    # Stop the pipeline
    pipeline.stop()


if __name__ == '__main__':
  app.run(main)
