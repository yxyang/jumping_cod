"""Policy that only takes in proprioceptive information (no perception)"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class Memory(torch.nn.Module):
  def __init__(self, input_size, type='gru', num_layers=1, hidden_size=256):
    super().__init__()
    # RNN
    rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
    self.rnn = rnn_cls(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers)
    self.hidden_states = None

  def forward(self, input, hidden_states=None):
    if len(input.shape) == 3:
      # Batch mode during training
      out, _ = self.rnn(input, hidden_states)
    else:
      # inference mode (collection): use hidden states of last step
      out, self.hidden_states = self.rnn(input.unsqueeze(0),
                                         self.hidden_states)
      out = out.squeeze(0)
    return out

  def reset(self, dones=None):
    if self.hidden_states is not None:
      # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
      for hidden_state in self.hidden_states:
        hidden_state[..., dones, :] = 0.0


class BasePolicyRecurrent(nn.Module):
  """Recurrent Policy"""
  def __init__(self):
    super().__init__()

  def forward(self,
              proprioceptive_state,
              height_map,
              depth_image,
              hidden_states=None):
    raise NotImplementedError()

  def reset(self, dones=None):
    raise NotImplementedError()

  def train_on_data(self, replay_buffer, batch_size: int, num_steps: int):
    self.train()
    optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')

    num_epoches = int(
        num_steps / max(replay_buffer.num_trajs / batch_size, 1)) + 1
    pbar = tqdm(range(num_epoches), desc="Training")
    for epoch in pbar:
      losses = []
      dataloader = replay_buffer.to_recurrent_generator(batch_size=batch_size)
      for batch in dataloader:
        optimizer.zero_grad()
        output = self.forward(batch['proprioceptive_states'],
                              batch['height_maps'], batch['depth_imgs'])
        loss = criterion(output, batch['actions'])
        loss = torch.sum(loss, dim=-1) / batch['actions'].shape[-1]
        loss = (loss * batch['masks']).sum() / batch['masks'].sum()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

      pbar.set_postfix({f"Avg Loss": f"{np.mean(losses):.4f}"})
    return np.mean(losses)

  def save(self, model_dir):
    torch.save(self.state_dict(), model_dir)
    print(f"Policy saved to: {model_dir}")

  def load(self, model_dir):
    self.load_state_dict(torch.load(model_dir, map_location="cpu"))
