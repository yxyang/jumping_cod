"""Generates different terrains for IsaacGym."""
import enum
from typing import Sequence

import ml_collections
import numpy as np
import scipy
import torch

from src.utils.torch_utils import to_torch


class GenerationMethod(enum.Enum):
  UNIFORM_RANDOM = 0
  CURRICULUM = 1
  SELECTED = 2


def select_terrain_random(names: Sequence[str], proportions: Sequence[float]):
  proportions = np.array(proportions).copy()
  proportions /= np.sum(proportions)
  return np.random.choice(names, p=proportions)


def select_terrain(names: Sequence[str], proportions: Sequence[float],
                   probability: float):
  cum_sum = 0
  for idx in range(len(proportions)):
    cum_sum += proportions[idx]
    if cum_sum > probability:
      return names[idx]

  raise ValueError("Invalid probability encountered.")


def box_trimesh(
    size,  # float [3] for x, y, z axis length (in meter) under box frame
    center_position,  # float [3] position (in meter) in world frame
):
  vertices = np.empty((8, 3), dtype=np.float32)
  vertices[:] = center_position
  vertices[[0, 4, 2, 6], 0] -= size[0] / 2
  vertices[[1, 5, 3, 7], 0] += size[0] / 2
  vertices[[0, 1, 2, 3], 1] -= size[1] / 2
  vertices[[4, 5, 6, 7], 1] += size[1] / 2
  vertices[[2, 3, 6, 7], 2] -= size[2] / 2
  vertices[[0, 1, 4, 5], 2] += size[2] / 2

  triangles = -np.ones((12, 3), dtype=np.uint32)
  triangles[0] = [0, 2, 1]  #
  triangles[1] = [1, 2, 3]
  triangles[2] = [0, 4, 2]  #
  triangles[3] = [2, 4, 6]
  triangles[4] = [4, 5, 6]  #
  triangles[5] = [5, 7, 6]
  triangles[6] = [1, 3, 5]  #
  triangles[7] = [3, 7, 5]
  triangles[8] = [0, 1, 4]  #
  triangles[9] = [1, 5, 4]
  triangles[10] = [2, 6, 3]  #
  triangles[11] = [3, 6, 7]

  return [vertices, triangles]


def combine_trimeshes(*trimeshes):
  if len(trimeshes) > 2:
    return combine_trimeshes(trimeshes[0], combine_trimeshes(*trimeshes[1:]))

  if len(trimeshes) <= 1:
    return trimeshes

  # only two trimesh to combine
  trimesh_0, trimesh_1 = trimeshes
  if trimesh_0[1].shape[0] < trimesh_1[1].shape[0]:
    trimesh_0, trimesh_1 = trimesh_1, trimesh_0

  trimesh_1 = (trimesh_1[0], trimesh_1[1] + trimesh_0[0].shape[0])
  vertices = np.concatenate((trimesh_0[0], trimesh_1[0]), axis=0)
  triangles = np.concatenate((trimesh_0[1], trimesh_1[1]), axis=0)

  return [vertices, triangles]


def move_trimesh(trimesh, move: np.ndarray):
  """ inplace operation """
  trimesh[0] += move


class Terrain:
  """Generates terrains as trimesh or heightfields."""
  def __init__(self,
               config: ml_collections.ConfigDict,
               device: str = 'cuda',
               random_seed=0):
    from isaacgym import terrain_utils
    self._terrain_utils = terrain_utils
    self._config = config
    self._device = device
    np.random.seed(random_seed)

    # Normalize proportions:
    sum_proportions = np.sum(self._config.terrain_proportions.values())
    self._terrain_names, self._terrain_proportions = [], []
    for terrain_name in sorted(self._config.terrain_proportions):
      self._terrain_names.append(terrain_name)
      self._terrain_proportions.append(
          self._config.terrain_proportions[terrain_name] / sum_proportions)

    # Initialize heightfield
    self._width_per_env_pixels = int(self._config.terrain_width /
                                     self._config.horizontal_scale)
    self._length_per_env_pixels = int(self._config.terrain_length /
                                      self._config.horizontal_scale)
    self._border_pixels = int(self._config.border_size /
                              self._config.horizontal_scale)
    self._total_cols = (self._config.num_cols * self._length_per_env_pixels +
                        2 * self._border_pixels)
    self._total_rows = (self._config.num_rows * self._width_per_env_pixels +
                        2 * self._border_pixels)

    # Initialize complete terrain map
    self._height_field_raw = np.zeros((self._total_rows, self._total_cols),
                                      dtype=np.int16)
    self._num_sub_terrains = self._config.num_rows * self._config.num_cols
    self._env_origins = np.zeros(
        (self._config.num_rows, self._config.num_cols, 3))
    self._additional_trimeshes = []

    self._make_all_terrains()
    self._env_origins = to_torch(self._env_origins, device=self._device)
    self._height_field_torch = to_torch(self._height_field_raw,
                                        device=self._device)
    if self._config.type == "trimesh":
      self.vertices, self.triangles = self._terrain_utils.convert_heightfield_to_trimesh(
          self._height_field_raw, self._config.horizontal_scale,
          self._config.vertical_scale, self._config.slope_threshold)

      # test_mesh = box_trimesh([0.05, 1, 0.02], [20, 20, -1.4])
      # import pdb
      # pdb.set_trace()
      if self._additional_trimeshes:
        self.vertices, self.triangles = combine_trimeshes(
            [self.vertices, self.triangles], self._additional_trimeshes)

  def _make_all_terrains(self):
    for j in range(self._config.num_cols):
      for i in range(self._config.num_rows):
        if self._config.generation_method == GenerationMethod.UNIFORM_RANDOM:
          terrain_type = select_terrain_random(self._terrain_names,
                                               self._terrain_proportions)
          difficulty = np.random.uniform(0., 1.)
        elif self._config.generation_method == GenerationMethod.CURRICULUM:
          terrain_type = select_terrain(self._terrain_names,
                                        self._terrain_proportions,
                                        j / self._config.num_cols)
          difficulty = i / self._config.num_rows
        else:
          terrain_type = self._config.selected_type
          difficulty = self._config.selected_difficulty
        subterrain, new_additional_trimeshes = self._make_sub_terrain(
            terrain_type, difficulty)
        self._add_subterrain_to_map(subterrain, i, j)
        if new_additional_trimeshes:
          move_trimesh(
              new_additional_trimeshes,
              np.array([
                  self._config.border_size + i * self._config.terrain_width,
                  self._config.border_size + j * self._config.terrain_length,
                  0,
              ]))
          if self._additional_trimeshes == []:
            self._additional_trimeshes = new_additional_trimeshes
          else:
            self._additional_trimeshes = combine_trimeshes(
                self._additional_trimeshes, new_additional_trimeshes)

  def _make_sub_terrain(self, terrain_type: str, difficulty: float):
    terrain = self._terrain_utils.SubTerrain(
        "terrain",
        width=self._width_per_env_pixels,
        length=self._length_per_env_pixels,
        vertical_scale=self._config.vertical_scale,
        horizontal_scale=self._config.horizontal_scale)
    additional_trimeshes = None

    if terrain_type == 'slope_smooth':
      slope = difficulty * 0.4 * 0.
      if np.random.uniform() > 0.5:
        slope *= -1  # Upward or downward slope
      self._terrain_utils.pyramid_sloped_terrain(terrain,
                                                 slope=slope,
                                                 platform_size=3.)
    elif terrain_type == 'slope_rough':
      slope = difficulty * 0.4
      if np.random.uniform() > 0.5:
        slope *= -1  # Upward or downward slope
      self._terrain_utils.pyramid_sloped_terrain(terrain,
                                                 slope=slope,
                                                 platform_size=3.)
      self._terrain_utils.random_uniform_terrain(terrain,
                                                 min_height=-0.01,
                                                 max_height=0.0,
                                                 step=0.005,
                                                 downsampled_scale=0.2)
    elif terrain_type == 'stair':
      step_height = 0.05 + 0.25 * difficulty
      if np.random.uniform() >= 0.:
        step_height *= -1

      if self._config.get("randomize_step_width", False):
        step_width = np.random.uniform(0.23, 0.37)
      else:
        step_width = 0.25
      additional_trimeshes = pyramid_stairs_terrain(
          terrain,
          step_width=step_width,
          step_height=step_height,
          platform_size=3.5,
          randomize_steps=self._config.get('randomize_steps', False))
    elif terrain_type == 'obstacles':
      obstacle_height = 0.05 + difficulty * 0.2
      self._terrain_utils.discrete_obstacles_terrain(
          terrain,
          max_height=obstacle_height,
          min_size=1,
          max_size=2,
          num_rects=20,
          platform_size=3.5)
    elif terrain_type == 'stepping_stones':
      stepping_stone_size = 0.5#np.random.uniform(0.1, 1)#.5 * (1.18 - difficulty)
      stone_distance = .6 * difficulty
      stepping_stones_terrain(terrain,
                              stone_size=stepping_stone_size,
                              stone_distance=stone_distance,
                              max_height=0.,
                              platform_size=5.,
                              depth=-0.4)
    elif terrain_type == 'gap':
      gap_size = 0.01 + 1. * difficulty
      gap_depth = 0.4
      gap_terrain(terrain,
                  gap_size=gap_size,
                  gap_depth=gap_depth,
                  platform_size=3.5)
    elif terrain_type == 'hurdle':
      gap_size = np.random.uniform(0.3, 0.7)
      gap_depth = -0.4 * difficulty - 0.01
      gap_terrain(terrain,
                  gap_size=gap_size,
                  gap_depth=gap_depth,
                  platform_size=3.5)
    elif terrain_type == 'pit':
      pit_depth = self._config.get(
          'pit_depth_min',
          0.) + (self._config.get('pit_depth_max', 0.7) -
                 self._config.get('pit_depth_min', 0.)) * difficulty
      pit_terrain(terrain,
                  depth=pit_depth,
                  platform_width=self._config.get('pit_platform_width', 3.5),
                  platform_length=self._config.get('pit_platform_length', 3.5))
    elif terrain_type == 'inv_pit':
      pit_depth = self._config.get(
          'pit_depth_min',
          0.) + (self._config.get('pit_depth_max', 0.7) -
                 self._config.get('pit_depth_min', 0.)) * difficulty
      pit_terrain(terrain,
                  depth=-pit_depth,
                  platform_width=self._config.get('pit_platform_width', 3.5),
                  platform_length=self._config.get('pit_platform_length', 3.5))
    else:
      raise ValueError("Unknown terrain type: {}".format(terrain_type))

    return terrain, additional_trimeshes

  def _add_subterrain_to_map(self, terrain: np.ndarray, row: int, col: int):
    i = row
    j = col
    # map coordinate system
    start_x = self._border_pixels + i * self._width_per_env_pixels
    end_x = self._border_pixels + (i + 1) * self._width_per_env_pixels
    start_y = self._border_pixels + j * self._length_per_env_pixels
    end_y = self._border_pixels + (j + 1) * self._length_per_env_pixels
    self._height_field_raw[start_x:end_x,
                           start_y:end_y] = terrain.height_field_raw

    env_origin_x = (i + 0.5) * self._config.terrain_width
    env_origin_y = (j + 0.5) * self._config.terrain_length
    x1 = int((self._config.terrain_width / 2. - 1) / terrain.horizontal_scale)
    x2 = int((self._config.terrain_width / 2. + 1) / terrain.horizontal_scale)
    y1 = int((self._config.terrain_length / 2. - 1) / terrain.horizontal_scale)
    y2 = int((self._config.terrain_length / 2. + 1) / terrain.horizontal_scale)
    env_origin_z = np.min(
        terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
    self._env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

  def get_terrain_height_at(self, xy_position: torch.Tensor):
    base_xy_in_hf = (xy_position[:, :2] +
                     self._config.border_size) / self._config.horizontal_scale
    base_idx = base_xy_in_hf.long()

    x, y = base_xy_in_hf[:, 0], base_xy_in_hf[:, 1]

    x1, y1 = base_idx[:, 0], base_idx[:, 1]

    x1 = torch.clip(x1, 0, self._height_field_torch.shape[0] - 2)
    y1 = torch.clip(y1, 0, self._height_field_torch.shape[1] - 2)

    x2 = x1 + 1
    y2 = y1 + 1
    # Perform bilinear interpolation
    # terrain_height = (self._height_field_torch[x1, y1] * (x2 - x) * (y2 - y) +
    #                   self._height_field_torch[x2, y1] * (x - x1) * (y2 - y) +
    #                   self._height_field_torch[x1, y2] * (x2 - x) * (y - y1) +
    #                   self._height_field_torch[x2, y2] * (x - x1) * (y - y1))
    terrain_height = torch.min(self._height_field_torch[x1, y1],
                               self._height_field_torch[x1, y2])
    terrain_height = torch.min(terrain_height, self._height_field_torch[x2,
                                                                        y1])
    terrain_height *= self._config.vertical_scale
    return terrain_height

  @property
  def env_origins(self):
    return self._env_origins

  @property
  def total_cols(self):
    return self._total_cols

  @property
  def total_rows(self):
    return self._total_rows

  @property
  def height_samples(self):
    return self._height_field_raw

  @property
  def config(self):
    return self._config


def pyramid_stairs_terrain(terrain,
                           step_width,
                           step_height,
                           platform_size=1.,
                           edge_width=0.05,
                           edge_height=0.02,
                           randomize_steps=False,
                           margin_size=0.75):
  """
    Generate stairs

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
  # switch parameters to discrete units
  step_width = int(step_width / terrain.horizontal_scale)
  step_height = int(step_height / terrain.vertical_scale)
  platform_size = int(platform_size / terrain.horizontal_scale) + 1
  margin_size = int(margin_size / terrain.horizontal_scale)
  # stair_edges = [box_trimesh([0.05, 1, 0.02], [10, 10, -4.4])]
  stair_edges = []

  height = 0
  start_x = margin_size
  stop_x = terrain.width - margin_size
  start_y = margin_size
  stop_y = terrain.length - margin_size
  next_step_width = step_width
  if randomize_steps:
    next_step_width += np.random.choice([-1, 0, 1])  # 5cm
  while (stop_x - start_x) > platform_size and (stop_y -
                                                start_y) > platform_size:
    curr_step_width = next_step_width
    next_step_width = step_width
    if randomize_steps:
      next_step_width += np.random.choice([-1, 0, 1])  # 5cm
    curr_step_height = step_height
    if randomize_steps:
      curr_step_height += np.random.randint(-5, 6)  # 2.5cm
    start_x += curr_step_width
    stop_x -= curr_step_width
    start_y += curr_step_width
    stop_y -= curr_step_width
    height += curr_step_height
    if np.random.random() > 10-0.5:
      stair_edges.append(
          box_trimesh([
              edge_width,
              (stop_y - start_y) * terrain.horizontal_scale,
              edge_height,
          ], [
              (stop_x - next_step_width) * terrain.horizontal_scale -
              edge_width / 2,
              terrain.length / 2 * terrain.horizontal_scale,
              height * terrain.vertical_scale - edge_height / 2,
          ]))

    terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height

  return combine_trimeshes(*stair_edges)


def gap_terrain(terrain, gap_size, gap_depth, platform_size=1.):
  gap_size = int(gap_size / terrain.horizontal_scale)
  gap_depth = int(gap_depth / terrain.vertical_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)

  center_x = terrain.width // 2
  center_y = terrain.length // 2
  x1 = platform_size // 2
  x2 = x1 + gap_size
  y1 = platform_size // 2
  y2 = y1 + gap_size

  terrain.height_field_raw[center_x - x2:center_x + x2,
                           center_y - y2:center_y + y2] = -gap_depth
  terrain.height_field_raw[center_x - x1:center_x + x1,
                           center_y - y1:center_y + y1] = 0


def pit_terrain(terrain, depth, platform_width=1., platform_length=1.):
  depth = int(depth / terrain.vertical_scale)
  platform_width = int(platform_width / terrain.horizontal_scale / 2)
  platform_length = int(platform_length / terrain.horizontal_scale / 2)
  x1 = terrain.width // 2 - platform_width
  x2 = terrain.width // 2 + platform_width
  y1 = terrain.length // 2 - platform_length
  y2 = terrain.length // 2 + platform_length
  terrain.height_field_raw[x1:x2, y1:y2] = -depth


def stepping_stones_terrain(terrain,
                            stone_size,
                            stone_distance,
                            max_height,
                            platform_size=1.,
                            depth=-10):
  """
    Generate a 1D stepping stones terrain

    Parameters:
        terrain (terrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """

  stone_size = int(stone_size / terrain.horizontal_scale)
  stone_distance = int(stone_distance / terrain.horizontal_scale)
  max_height = int(max_height / terrain.vertical_scale)
  platform_size = int(platform_size / terrain.horizontal_scale)
  height_range = np.arange(-max_height, max_height + 1, step=1)
  # stair_edges = [box_trimesh([0.05, 1, 0.02], [10, 10, -4.4])]

  height = 0
  start_x = 0
  stop_x = terrain.width
  start_y = 0
  stop_y = terrain.length
  while (stop_x - start_x) > platform_size and (stop_y -
                                                start_y) > platform_size:
    start_x += stone_size
    stop_x -= stone_size
    start_y += stone_size
    stop_y -= stone_size
    height = int(depth / terrain.vertical_scale)
    terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height

    start_x += stone_distance
    stop_x -= stone_distance
    start_y += stone_distance
    stop_y -= stone_distance
    terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = 0
