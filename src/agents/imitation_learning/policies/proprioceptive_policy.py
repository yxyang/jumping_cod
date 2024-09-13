"""Policy that only takes in proprioceptive information (no perception)"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.agents.imitation_learning.policies import base_policy


class ProprioceptivePolicy(base_policy.BasePolicy):
  """MLP Policy"""
  def __init__(self,
               dim_state: int = 28,
               dim_action: int = 16,
               num_hidden=3,
               dim_hidden=256,
               activation=F.elu):
    super().__init__()
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(dim_state, dim_hidden))
    for _ in range(num_hidden - 1):
      self.layers.append(nn.Linear(dim_hidden, dim_hidden))
    self.layers.append(nn.Linear(dim_hidden, dim_action))
    self.activation = activation

  def forward(self, proprioceptive_state, height_map, depth_image):
    del height_map  # unused
    del depth_image  # unused
    curr = proprioceptive_state
    for layer in self.layers[:-1]:
      curr = layer(curr)
      curr = self.activation(curr)

    curr = self.layers[-1](curr)
    return curr
