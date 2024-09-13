"""Actor classes."""
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

from src.agents.imitation_learning.policies import base_policy_recurrent
from src.agents.imitation_learning.policies import depth_policy


class DepthPolicyRecurrent(base_policy_recurrent.BasePolicyRecurrent):
  """MLP Policy"""
  def __init__(self,
               dim_state: int = 27,
               dim_action: int = 22,
               camera_vertical_res: int = 48,
               camera_horizontal_res: int = 16,
               rnn_type: str = 'gru',
               rnn_hidden_size=256,
               rnn_num_layers=1,
               num_hidden=3,
               dim_hidden=256,
               activation=F.elu):
    super().__init__()
    self.image_embedding = depth_policy.Conv2dModel()
    self.dim_embedding = self.image_embedding.conv_out_size(
        camera_vertical_res, camera_horizontal_res)

    self.memory = base_policy_recurrent.Memory(dim_state + self.dim_embedding,
                                               type=rnn_type,
                                               num_layers=rnn_num_layers,
                                               hidden_size=rnn_hidden_size)
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(rnn_hidden_size, dim_hidden))
    for _ in range(num_hidden - 1):
      self.layers.append(nn.Linear(dim_hidden, dim_hidden))
    self.layers.append(nn.Linear(dim_hidden, dim_action))
    self.activation = activation

  def forward(self,
              proprioceptive_state,
              height_map,
              depth_image,
              hidden_states=None):
    del height_map  # unused
    depth_image_normalized = -torch.nan_to_num(depth_image, neginf=-3) / 3
    depth_image_normalized = depth_image_normalized.clip(min=0, max=1)

    original_shape = depth_image.shape  # Length x Batch x H x W
    cnn_input = depth_image_normalized.reshape(
        [-1, 1, original_shape[-2], original_shape[-1]])  # NxCxHxW
    depth_emb = self.image_embedding(cnn_input).reshape(
        original_shape[:-2] + (-1, ))  # Length x Batch x Embedding

    curr = torch.concatenate((proprioceptive_state, depth_emb), dim=-1)
    rnn_out = self.memory(curr, hidden_states=hidden_states)
    curr = rnn_out
    for layer in self.layers[:-1]:
      curr = layer(curr)
      curr = self.activation(curr)

    curr = self.layers[-1](curr)
    return curr

  def forward_and_return_embedding(self,
                                   proprioceptive_state,
                                   height_map,
                                   depth_image,
                                   hidden_states=None):
    del height_map  # unused
    depth_image_normalized = -torch.nan_to_num(depth_image, neginf=-3) / 3
    depth_image_normalized = depth_image_normalized.clip(min=0, max=1)

    original_shape = depth_image.shape  # Length x Batch x H x W
    cnn_input = depth_image_normalized.reshape(
        [-1, 1, original_shape[-2], original_shape[-1]])  # NxCxHxW
    depth_emb = self.image_embedding(cnn_input).reshape(
        original_shape[:-2] + (-1, ))  # Length x Batch x Embedding

    curr = torch.concatenate((proprioceptive_state, depth_emb), dim=-1)
    rnn_out = self.memory(curr, hidden_states=hidden_states)
    curr = rnn_out
    for layer in self.layers[:-1]:
      curr = layer(curr)
      curr = self.activation(curr)

    curr = self.layers[-1](curr)
    return curr, depth_emb

  def forward_with_embedding(self,
                             embedding,
                             proprioceptive_state,
                             hidden_states=None):
    rnn_input = torch.concatenate((proprioceptive_state, embedding), dim=-1)
    rnn_out = self.memory(rnn_input, hidden_states=hidden_states)
    curr = rnn_out
    for layer in self.layers[:-1]:
      curr = layer(curr)
      curr = self.activation(curr)

    curr = self.layers[-1](curr)
    return curr

  def get_hidden_states(self):
    return self.memory.hidden_states

  def reset(self, dones=None):
    self.memory.reset(dones)
