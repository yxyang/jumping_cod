"""Actor classes."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class Memory(torch.nn.Module):
  """Memory module for RNN."""
  def __init__(self,
               input_size,
               memory_type='gru',
               num_layers=1,
               hidden_size=256):
    super().__init__()
    # RNN
    rnn_cls = nn.GRU if memory_type.lower() == 'gru' else nn.LSTM
    self.rnn = rnn_cls(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers)
    self.hidden_states = None

  def forward(self, input_tensor, hidden_states=None):
    if len(input_tensor.shape) == 3:
      # Batch mode during training
      out, _ = self.rnn(input_tensor, hidden_states)
    else:
      # inference mode (collection): use hidden states of last step
      out, self.hidden_states = self.rnn(input_tensor.unsqueeze(0),
                                         self.hidden_states)
      out = out.squeeze(0)
    return out

  def reset(self, dones=None):
    if self.hidden_states is not None:
      # When the RNN is an LSTM, self.hidden_states_a is a list
      # with hidden_state and cell_state
      if dones is None:
        self.hidden_states = None
      else:
        for state in self.hidden_states:  # pytype: disable=attribute-error
          state[..., dones, :] = 0.0


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
      channels=(2, 4, 8),  #[16, 32, 32],
      kernel_sizes=(5, 4, 3),
      strides=(2, 1, 1),  #[2, 2, 1],
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
    in_channels = [in_channels] + list(channels)[:-1]
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

  def forward(self, input_tensor):
    """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
    return self.conv(input_tensor)

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


class LSTMHeightmapPredictor(nn.Module):
  """MLP Policy"""
  def __init__(self,
               dim_output: int,
               vertical_res: int = 48,
               horizontal_res: int = 64,
               rnn_type: str = 'gru',
               rnn_hidden_size=256,
               rnn_num_layers=1,
               num_hidden=3,
               dim_hidden=256,
               activation=F.elu):
    super().__init__()
    self.image_embedding = Conv2dModel()
    self.dim_embedding = self.image_embedding.conv_out_size(
        vertical_res, horizontal_res)
    self.dim_base_state = 8
    self.dim_foot_position = 8

    self.memory = Memory(self.dim_embedding + self.dim_base_state +
                         self.dim_foot_position,
                         memory_type=rnn_type,
                         num_layers=rnn_num_layers,
                         hidden_size=rnn_hidden_size)
    self.layers = nn.ModuleList()
    self.layers.append(nn.Linear(rnn_hidden_size, dim_hidden))
    for _ in range(num_hidden - 1):
      self.layers.append(nn.Linear(dim_hidden, dim_hidden))
    self.layers.append(nn.Linear(dim_hidden, dim_output))
    self.activation = activation

  def forward(self,
              base_state,
              foot_positions,
              depth_image,
              hidden_states=None):
    depth_image_normalized = -torch.nan_to_num(depth_image, neginf=-3) / 3
    depth_image_normalized = depth_image_normalized.clip(min=0, max=1)
    return self.forward_normalized(base_state, foot_positions,
                                   depth_image_normalized, hidden_states)

  def forward_normalized(self,
                         base_state,
                         foot_positions,
                         depth_image_normalized,
                         hidden_states=None):
    original_shape = depth_image_normalized.shape  # Length x Batch x H x W
    cnn_input = depth_image_normalized.reshape(
        [-1, 1, original_shape[-2], original_shape[-1]])  # NxCxHxW
    depth_emb = self.image_embedding(cnn_input).reshape(
        original_shape[:-2] + (-1, ))  # Length x Batch x Embedding

    curr = torch.concatenate((
        base_state,
        foot_positions,
        depth_emb,
    ), dim=-1)
    rnn_out = self.memory(curr, hidden_states=hidden_states)
    curr = rnn_out
    for layer in self.layers[:-1]:
      curr = layer(curr)
      curr = self.activation(curr)
    heightmap = self.layers[-1](curr)
    return heightmap

  def get_hidden_states(self):
    return self.memory.hidden_states

  def reset(self, dones=None):
    self.memory.reset(dones)

  def train_on_data(self, replay_buffer, batch_size: int, num_steps: int):
    self.train()
    optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')

    num_epoches = int(
        num_steps / max(replay_buffer.num_trajs / batch_size, 1)) + 1
    pbar = tqdm(range(num_epoches), desc="Training")
    for _ in pbar:
      losses = []
      dataloader = replay_buffer.to_recurrent_generator(batch_size=batch_size)
      for batch in dataloader:
        optimizer.zero_grad()
        output = self.forward(batch['base_states'], batch['foot_positions'],
                              batch['depth_imgs'])
        loss = criterion(output, batch['height_maps'])
        loss = torch.sum(loss, dim=-1) / batch['height_maps'].shape[-1]
        loss = (loss * batch['masks']).sum() / batch['masks'].sum()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

      pbar.set_postfix({f"Avg Loss": f"{np.mean(losses):.4f}"})
    return np.mean(losses)

  def save(self, model_dir):
    torch.save(self.state_dict(), model_dir)  # pylint: disable=missing-kwoa
    print(f"Predictor saved to: {model_dir}")

  def load(self, model_dir):
    self.load_state_dict(torch.load(model_dir, map_location="cpu"))
