"""Replay Buffer to store collected trajectories."""

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class ReplayBuffer:
  """Replay buffer to store collected trajectories."""
  def __init__(self, env, device: str):
    self._env = env
    self._device = device
    self._proprioceptive_states = []
    self._height_maps = []
    self._depth_imgs = []
    self._actions = []
    self._steps_count = 0
    self._reward_sums = torch.zeros(env.num_envs, device=self._device)
    self._num_envs = env.num_envs

    self._env.reset()

  def _record_new_traj(self, proprioceptive_states, height_maps, depth_imgs,
                       actions, start_idx, end_idx, env_id):
    # Input: batched states: list of [num_envs x dim_state]
    # Select the env_id, from start_idx to end_idx
    if start_idx == end_idx:
      return

    self._steps_count += (end_idx - start_idx)
    self._proprioceptive_states.append(
        torch.stack([
            proprioceptive_state[env_id] for proprioceptive_state in
            proprioceptive_states[start_idx:end_idx]
        ],
                    dim=0))
    self._height_maps.append(
        torch.stack([
            height_map[env_id] for height_map in height_maps[start_idx:end_idx]
        ],
                    dim=0))
    self._depth_imgs.append(
        torch.stack(
            [depth_img[env_id] for depth_img in depth_imgs[start_idx:end_idx]],
            dim=0))
    self._actions.append(
        torch.stack([action[env_id] for action in actions[start_idx:end_idx]],
                    dim=0))

  def collect_data(self,
                   acting_policy,
                   teacher_policy,
                   num_steps: int,
                   initial_rollout: bool = False):
    self._env.reset()
    acting_policy.reset()
    teacher_policy.reset()
    if not initial_rollout:
      acting_policy.eval()
    steps_count = 0
    infos = []
    pbar = tqdm(total=num_steps, desc="Collecting Data", leave=True)
    state = self._env.get_observations()
    # Initialize trajectory buffers
    proprioceptive_states = []
    height_maps = []
    depth_imgs = []
    actions = []
    sum_rewards = []
    cycle_counts = []

    start_indices = torch.zeros(self._num_envs,
                                device=self._device,
                                dtype=torch.int64)
    start_count = self._steps_count
    with torch.no_grad():
      # while steps_count * self._num_envs < num_steps:
      while self._steps_count - start_count < num_steps:
        # pbar.update(self._num_envs)
        steps_count += 1
        curr_imgs = []
        for env_id in range(self._num_envs):
          curr_imgs.append(
              to_torch(self._env.robot.get_camera_image(env_id, mode="depth"),
                       device=self._device))
        curr_imgs = torch.stack(curr_imgs, dim=0).clone()

        if initial_rollout:
          # The PPO policy takes in a different state space
          action = acting_policy.act_inference(state)
        else:
          proprioceptive_state = self._env.get_proprioceptive_observation()
          height_map = self._env.get_heights(ground_truth=True)
          action = acting_policy.forward(proprioceptive_state, height_map,
                                         curr_imgs)
          action = action.clip(min=self._env.action_space[0],
                               max=self._env.action_space[1])
        teacher_action = teacher_policy.act_inference(state)

        proprioceptive_states.append(
            self._env.get_proprioceptive_observation().clone())
        height_maps.append(self._env.get_heights(ground_truth=True).clone())
        depth_imgs.append(curr_imgs)
        actions.append(teacher_action.clone())

        state, _, reward, dones, info = self._env.step(action)
        self._reward_sums += reward.clone()
        if dones.any():
          acting_policy.reset(dones)
          teacher_policy.reset(dones)
          cycle_counts.append(info["episode"]["cycle_count"].cpu().numpy())
          done_idx = dones.nonzero(as_tuple=False).flatten()
          sum_rewards.extend(list(self._reward_sums[done_idx].cpu().numpy()))
          self._reward_sums[done_idx] = 0
          for env_id in done_idx:
            self._record_new_traj(proprioceptive_states, height_maps,
                                  depth_imgs, actions,
                                  start_indices[env_id].item(), steps_count,
                                  env_id)
            pbar.update(steps_count - start_indices[env_id].item())
            start_indices[env_id] = steps_count

      # Remaining trajs at the end of the rollout
      # for env_id in range(self._num_envs):
      #   self._record_new_traj(proprioceptive_states, height_maps, depth_imgs,
      #                         actions,
      #                         start_indices[env_id].item(), steps_count,
      #                         env_id)
    pbar.close()
    return sum_rewards, cycle_counts, infos

  def to_dataloader(self, batch_size: int):
    dataset = TensorDataset(
        torch.concatenate(self._proprioceptive_states, dim=0),
        torch.concatenate(self._height_maps, dim=0),
        torch.concatenate(self._depth_imgs, dim=0),
        torch.concatenate(self._actions, dim=0))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

  def _prepare_padded_sequence(self, traj_indices):
    traj_lengths = [
        self._proprioceptive_states[idx].shape[0] for idx in traj_indices
    ]
    max_length = max(traj_lengths)
    num_trajs = len(traj_indices)

    proprioceptive_states = torch.zeros(
        (max_length, num_trajs, self._proprioceptive_states[0].shape[1]),
        device=self._device)
    height_maps = torch.zeros(
        (max_length, num_trajs, self._height_maps[0].shape[1]),
        device=self._device)
    depth_imgs = torch.zeros(
        (max_length, num_trajs, self._depth_imgs[0].shape[1],
         self._depth_imgs[0].shape[2]),
        device=self._device)
    actions = torch.zeros((max_length, num_trajs, self._actions[0].shape[1]),
                          device=self._device)
    masks = torch.zeros((max_length, num_trajs),
                        dtype=torch.bool,
                        device=self._device)
    for output_idx, traj_idx in enumerate(traj_indices):
      proprioceptive_states[:traj_lengths[output_idx],
                            output_idx] = self._proprioceptive_states[traj_idx]
      height_maps[:traj_lengths[output_idx],
                  output_idx] = self._height_maps[traj_idx]
      depth_imgs[:traj_lengths[output_idx],
                 output_idx] = self._depth_imgs[traj_idx]
      actions[:traj_lengths[output_idx], output_idx] = self._actions[traj_idx]
      masks[:traj_lengths[output_idx], output_idx] = 1

    return dict(proprioceptive_states=proprioceptive_states,
                height_maps=height_maps,
                depth_imgs=depth_imgs,
                actions=actions,
                masks=masks)

  def to_recurrent_generator(self, batch_size: int):
    num_trajs = len(self._proprioceptive_states)
    traj_indices = np.arange(num_trajs)
    traj_indices = np.random.permutation(traj_indices)
    for start_idx in range(0, num_trajs, batch_size):
      end_idx = np.minimum(start_idx + batch_size, num_trajs)
      yield self._prepare_padded_sequence(traj_indices[start_idx:end_idx])

  def save(self, rb_dir):
    torch.save(
        dict(proprioceptive_states=self._proprioceptive_states,
             height_maps=self._height_maps,
             depth_imgs=self._depth_imgs,
             actions=self._actions), rb_dir)
    print(f"Replay buffer saved to: {rb_dir}")

  def load(self, rb_dir):
    traj = torch.load(rb_dir)
    self._proprioceptive_states = traj["proprioceptive_states"]
    self._height_maps = traj["height_maps"]
    self._depth_imgs = traj["depth_imgs"]
    self._actions = traj["actions"]

  @property
  def num_trajs(self):
    return len(self._proprioceptive_states)
