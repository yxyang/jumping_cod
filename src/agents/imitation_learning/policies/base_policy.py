"""Policy that only takes in proprioceptive information (no perception)"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class BasePolicy(nn.Module):
  """MLP Policy"""
  def __init__(self):
    super().__init__()

  def forward(self, proprioceptive_state, height_map, depth_image):
    raise NotImplementedError()

  def train_on_data(self, replay_buffer, batch_size: int, num_steps: int):
    self.train()
    optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    dataloader = replay_buffer.to_dataloader(batch_size=batch_size)

    num_epoches = int(num_steps / len(dataloader)) + 1
    pbar = tqdm(range(num_epoches), desc="Training")
    # if num_epoches <= 40:
    #   import pdb
    #   pdb.set_trace()
    for epoch in pbar:
      losses = []
      for state, heightmap, depth_image, actions in dataloader:
        optimizer.zero_grad()
        output = self.forward(state, heightmap, depth_image)
        loss = criterion(output, actions)
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

  def reset(self, dones=None):
    # No reset needed for memoryless policy
    del dones  # unused
    pass
