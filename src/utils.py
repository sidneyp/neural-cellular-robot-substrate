"""
Utils
"""
import tensorflow as tf
import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import json
from types import SimpleNamespace
import os
import csv

def get_weights_info(weights):
  weight_shape_list = []
  for layer in weights:
    weight_shape_list.append(tf.shape(layer))

  weight_amount_list = [tf.reduce_prod(w_shape)\
                        for w_shape in weight_shape_list]
  weight_amount = tf.reduce_sum(weight_amount_list)

  return weight_shape_list, weight_amount_list, weight_amount


def get_model_weights(flat_weights, weight_amount_list, weight_shape_list):
  split_weight = tf.split(flat_weights, weight_amount_list)
  return [tf.reshape(split_weight[i], weight_shape_list[i])\
          for i in tf.range(len(weight_shape_list))]

def get_flat_weights(weights):
  flat_weights = []
  for layer in weights:
    flat_weights.extend(list(layer.numpy().flatten()))
  return flat_weights

# Based on: https://github.com/google-research/self-organising-systems
class CAPool:
  def __init__(self, grids):
    self.size = len(grids)
    self.grids = grids
    self.is_seed = np.full(self.size, True)
    self.current_idx = None

  def sample(self, n):
    self.current_idx = np.random.choice(self.size, n, False)
    return self.grids[self.current_idx], self.is_seed[self.current_idx]

  def commit(self, batch_grids, batch_is_seed):
    self.grids[self.current_idx] = batch_grids
    self.is_seed[self.current_idx] = batch_is_seed
    self.current_idx = None

# Based on: https://github.com/google-research/self-organising-systems
class EnvCAPool(CAPool):
  def __init__(self, envs, grids):
    super().__init__(grids)
    assert self.size == len(envs), "EnvCAPool: envs and grids must have the same lenght!"
    self.envs = np.asarray(envs)

  def sample(self, n):
    self.current_idx = np.random.choice(self.size, n, False)
    return self.envs[self.current_idx], self.grids[self.current_idx],\
      self.is_seed[self.current_idx]

  def commit(self, batch_envs, batch_grids, batch_is_seed):
    self.envs[self.current_idx] = batch_envs
    self.grids[self.current_idx] = batch_grids
    self.is_seed[self.current_idx] = batch_is_seed
    self.current_idx = None

# Code from: https://github.com/google-research/self-organising-systems
class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()

# Code from: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def fig2array(fig):
  # If not drawn
  fig.canvas.draw()
  # Now we can save it to a numpy array.
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data

# Based on: https://stackoverflow.com/questions/57541522/save-load-a-dictionary
class ArgsIO:
  def __init__(self, filename):
    self.filename = filename

  def save_json(self):
    print(self.__dict__)
    with open(self.filename, 'w') as f:
      f.write(json.dumps(self.__dict__))

  def load_json(self):
    with open(self.filename, 'r') as f:
      dictionary = json.loads(f.read())
    return SimpleNamespace(**dictionary)


def manual_body_grid(grid_type, env_amount):
  body_grid = None
  if grid_type == 1:
    body_grid = np.array([[2,1,2], [3,1,3], [1,2,1]])
  elif grid_type == 2:
    body_grid = np.array([[3,2,3]])
  elif grid_type == 3:
    body_grid = np.array([[1,2,1,2,1],
                          [3,1,1,1,3],
                          [0,1,2,1,0],
                          [3,1,1,1,3],
                          [1,1,1,1,1]])
  else:
    print("Warning: Invalid manual body type.")

  manual_body_grid = None
  if body_grid is not None:
    body_grid_exp = np.expand_dims(body_grid, 0)
    manual_body_grid = np.repeat(body_grid_exp, env_amount, axis=0)

  return manual_body_grid

def manual_body_grid_carrier(grid_type, env_amount):
  body_grid = None
  if grid_type == 1:
    body_grid = np.array([[2,1,2], [3,1,3], [1,2,1]])
  else:
    print("Warning: Invalid manual body type.")

  manual_body_grid = None
  if body_grid is not None:
    body_grid_exp = np.expand_dims(body_grid, 0)
    manual_body_grid = np.repeat(body_grid_exp, env_amount, axis=0)

  return manual_body_grid


def calculate_build_body_steps(height, width):
  max_grid_size = max(height, width)
  return int((max_grid_size-np.floor(max_grid_size/2)-1)*5)

def save_generation(loss, features, body, gen, folder_path):
  csv_file_path = os.path.join(folder_path, "{:06d}.csv".format(gen))
  with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["features", "loss",
                     "body", "solution"])
    for i in range(len(loss)):
      writer.writerow([features[i],
                      loss[i],
                      body[i]])

def save_generation_with_solutions(solutions, loss, features, body, gen, folder_path):
  csv_file_path = os.path.join(folder_path, "{:06d}.csv".format(gen))
  with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["features", "loss",
                     "body", "solution"])
    for i in range(len(loss)):
      writer.writerow([features[i],
                      loss[i],
                      body[i],
                      list(solutions[i])])

def reorganize_obs(body_grid, sensor_n, obs):
  obs_array = np.asarray(obs)
  new_obs = np.array(obs)
  flat_bg = body_grid.flatten()
  idx_array = np.arange(np.prod(body_grid.shape))
  for batch_idx in range(len(obs)):
    sensor_number_per_type = []
    idx_array_per_type = []
    for i in range(2,sensor_n+2):
      idx_array_i = idx_array[flat_bg==i]
      sensor_number_per_type.append(len(idx_array_i))
      idx_array_per_type.append(idx_array_i)

    idx_array_concat = np.concatenate(idx_array_per_type, 0)
    idx_array_sort_idx = np.argsort(idx_array_concat)
    new_obs[batch_idx] = obs_array[batch_idx, idx_array_sort_idx]

  return new_obs

