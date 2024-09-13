"""Actor classes."""
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

from src.agents.imitation_learning.policies import base_policy


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
  """
    Returns output H, W after convolution/pooling on input H, W.
    """
  kh, kw = kernel_size if isinstance(kernel_size,
                                     tuple) else (kernel_size, ) * 2
  sh, sw = stride if isinstance(stride, tuple) else (stride, ) * 2
  ph, pw = padding if isinstance(padding, tuple) else (padding, ) * 2
  d = dilation
  h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
  w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
  return h, w


class Conv2dModel(torch.nn.Module):
  """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """
  def __init__(
      self,
      in_channels=1,
      channels=[2, 4, 8],  #[16, 32, 32],
      kernel_sizes=[5, 4, 3],
      strides=[2, 1, 1],  #[2, 2, 1],
      paddings=None,
      nonlinearity=torch.nn.LeakyReLU,  # Module, not Functional.
      use_maxpool=True,  # if True: convs use stride 1, maxpool downsample.
      normlayer=None,  # If None, will not be used
  ):
    super().__init__()
    if paddings is None:
      paddings = [0 for _ in range(len(channels))]
    if isinstance(normlayer, str):
      normlayer = getattr(torch.nn, normlayer)
    assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
    in_channels = [in_channels] + channels[:-1]
    ones = [1 for _ in range(len(strides))]
    if use_maxpool:
      maxp_strides = strides
      strides = ones
    else:
      maxp_strides = ones
    conv_layers = [
        torch.nn.Conv2d(in_channels=ic,
                        out_channels=oc,
                        kernel_size=k,
                        stride=s,
                        padding=p)
        for (ic, oc, k, s,
             p) in zip(in_channels, channels, kernel_sizes, strides, paddings)
    ]
    sequence = list()
    for conv_layer, oc, maxp_stride in zip(conv_layers, channels,
                                           maxp_strides):
      if normlayer is not None:
        sequence.extend([conv_layer, normlayer(oc), nonlinearity()])
      else:
        sequence.extend([conv_layer, nonlinearity()])
      if maxp_stride > 1:
        sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
    self.conv = torch.nn.Sequential(*sequence)

  def forward(self, input):
    """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
    return self.conv(input)

  def conv_out_size(self, h, w):
    """Helper function ot return the output size for a given input shape,
      without actually performing a forward pass through the model."""
    for child in self.conv.children():
      try:
        h, w = conv2d_output_shape(h, w, child.kernel_size, child.stride,
                                   child.padding)
      except AttributeError:
        pass  # Not a conv or maxpool layer.
      try:
        c = child.out_channels
      except AttributeError:
        pass  # Not a conv layer.
    return h * w * c


class DepthPolicy(base_policy.BasePolicy):
  """MLP Policy"""
  def __init__(self,
               dim_state: int = 28,
               dim_action: int = 16,
               camera_vertical_res: int = 48,
               camera_horizontal_res: int = 64,
               num_hidden=3,
               dim_hidden=256,
               activation=F.elu):
    super().__init__()
    self.image_embedding = Conv2dModel()

    dim_embedding = self.image_embedding.conv_out_size(camera_vertical_res,
                                                       camera_horizontal_res)
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(dim_state + dim_embedding, dim_hidden))
    for _ in range(num_hidden - 1):
      self.layers.append(nn.Linear(dim_hidden, dim_hidden))
    self.layers.append(nn.Linear(dim_hidden, dim_action))
    self.activation = activation

  def forward(self, proprioceptive_state, height_map, depth_image):
    del height_map  # unused
    depth_image_normalized = -torch.nan_to_num(depth_image, neginf=-3) / 3
    depth_image_normalized = depth_image_normalized.clip(min=0, max=1)
    depth_image_normalized = depth_image_normalized.unsqueeze(1)
    depth_emb = self.image_embedding(depth_image_normalized).reshape(
        (proprioceptive_state.shape[0], -1))
    curr = torch.concatenate((proprioceptive_state, depth_emb), dim=-1)
    for layer in self.layers[:-1]:
      curr = layer(curr)
      curr = self.activation(curr)

    curr = self.layers[-1](curr)
    return curr
