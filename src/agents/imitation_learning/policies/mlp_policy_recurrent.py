"""Actor classes."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.agents.imitation_learning.policies import base_policy_recurrent


class MLPPolicyRecurrent(base_policy_recurrent.BasePolicyRecurrent):
  """MLP Policy"""
  def __init__(self,
               dim_state: int = 248,
               dim_action: int = 16,
               rnn_type: str = 'gru',
               rnn_hidden_size=256,
               rnn_num_layers=1,
               num_hidden=3,
               dim_hidden=256,
               activation=F.elu):
    super().__init__()
    self.memory = base_policy_recurrent.Memory(dim_state,
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
    del depth_image  # unused
    curr = torch.concatenate((proprioceptive_state, height_map), dim=-1)
    curr = self.memory(curr, hidden_states=hidden_states)
    for layer in self.layers[:-1]:
      curr = layer(curr)
      curr = self.activation(curr)

    curr = self.layers[-1](curr)
    return curr

  def get_hidden_states(self):
    return self.memory.hidden_states

  def reset(self, dones=None):
    self.memory.reset(dones)
