import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
tf.config.set_visible_devices([], 'GPU')

# Parameters
BODY_STATIC_N = 1
BODY_SENSOR_N = 1
BODY_ACTUATOR_N = 1
HIDDEN_CHANNEL_N = 6
BATCH_SIZE = 16
POOL_SIZE = BATCH_SIZE * 10
CELL_FIRE_RATE = 1.0#0.5
CA_STEPS_BUILD_BODY = (20,21)#(50,60)
CA_STEPS_PER_ACTION = (2,3)#(30,40)
ADD_NOISE = False #True
KERNEL_INIT = "zeros"
LAYER_1_FILTER_N = 30
LAYER_2_NEURON_N = 30
CLIP_VALUE = 5
ACT_BODY_LIMITATION = True
ONLY_LIMITED_CHANNELS = True
SEPARATED_INPUT_OUTPUT_CHANNELS = False
ADD_CONTROL_IDENTIFIER_CHANNEL = True
IS_BODY_LIMITED_CONTROL_ID = False

#@tf.function
def get_energy_mask(x):
  energy = x[:, :, :, :1]
  return tf.nn.max_pool2d(energy, 3, [1, 1, 1, 1], 'SAME') > 0.1

#@tf.function
def get_body_mask(x):
  return x[:, :, :, :1] > 0.1

class BodyBrainNCA(tf.keras.Model):
  def __init__(self, body_static_n=BODY_STATIC_N,
               body_sensor_n=BODY_SENSOR_N,
               body_actuator_n=BODY_ACTUATOR_N,
               hidden_channel_n=HIDDEN_CHANNEL_N,
               layer_1_filter_n=LAYER_1_FILTER_N,
               layer_2_neuron_n=LAYER_2_NEURON_N,
               fire_rate=CELL_FIRE_RATE,
               ca_steps_build_body=CA_STEPS_BUILD_BODY,
               ca_steps_per_action=CA_STEPS_PER_ACTION,
               clip_value=CLIP_VALUE,
               add_noise=ADD_NOISE,
               act_body_limitation=ACT_BODY_LIMITATION,
               only_limited_channels=ONLY_LIMITED_CHANNELS,
               separated_input_output_channels=SEPARATED_INPUT_OUTPUT_CHANNELS,
               add_control_identifier_channel=ADD_CONTROL_IDENTIFIER_CHANNEL,
               is_body_limited_control_id=IS_BODY_LIMITED_CONTROL_ID,
               kernel_init=KERNEL_INIT,
               **kwargs):
    super().__init__()
    self.body_channel_n = body_static_n + body_sensor_n + body_actuator_n
    self.body_static_n = body_static_n
    self.body_sensor_n = body_sensor_n
    self.body_actuator_n = body_actuator_n
    self.hidden_channel_n = hidden_channel_n

    self.separated_input_output_channels = separated_input_output_channels
    self.add_control_identifier_channel = add_control_identifier_channel
    self.is_body_limited_control_id = is_body_limited_control_id
    self.only_limited_channels = only_limited_channels

    self.energy_ctrl_channel_n = 2 if self.add_control_identifier_channel else 1

    # Total number of body, control identifier and energy channels.
    self.body_energy_channel_n = self.body_channel_n + self.energy_ctrl_channel_n

    self.io_channel_n = 2 if self.separated_input_output_channels else 1
    self.brain_channel_n = hidden_channel_n + self.io_channel_n

    # Total number of channels.
    self.channel_n = self.body_energy_channel_n + self.brain_channel_n

    self.limited_channel_n = \
      self.channel_n if self.only_limited_channels else self.body_energy_channel_n
    self.unlimited_channel_n = \
      0 if self.only_limited_channels else self.brain_channel_n

    self.fire_rate = fire_rate
    self.ca_steps_build_body = tuple([int(v) for v in ca_steps_build_body])
    self.ca_steps_per_action = tuple([int(v) for v in ca_steps_per_action])
    self.clip_value = clip_value
    self.add_noise = add_noise
    self.act_body_limitation = act_body_limitation
    self.layer_1_filter_n = layer_1_filter_n
    self.layer_2_neuron_n = layer_2_neuron_n

    if kernel_init == "zeros":
      kernel_initializer = tf.zeros_initializer
    elif kernel_init == "random":
      kernel_initializer = tf.initializers.GlorotUniform

    self.dmodel = tf.keras.Sequential([
          Conv2D(self.layer_1_filter_n, 3, activation=tf.nn.relu, padding="SAME"),
          Conv2D(self.layer_2_neuron_n, 1, activation=tf.nn.relu),
          Conv2D(self.channel_n, 1, activation=None,
                 kernel_initializer=kernel_initializer)
    ])

    self(tf.zeros([1, 3, 3, self.channel_n]))  # dummy calls to build the model

  def get_dict_args(self):
    nca_dict = dict(self.__dict__)
    del nca_dict["dmodel"]

    list_delete_key = []
    for k in nca_dict:
      if k[0] == "_":
        list_delete_key.append(k)

    for k in list_delete_key:
      del nca_dict[k]

    return nca_dict

  #@tf.function
  def apply_sensor_inputs(self, x, sensor_inputs):
    sensor_inputs_flat = []
    for s in sensor_inputs:
      sensor_inputs_flat.extend(s)

    energy, static, sensor, actuator, hidden, io =\
    tf.split(x,
             [self.energy_ctrl_channel_n, # energy (+control id)
              self.body_static_n,
              self.body_sensor_n,
              self.body_actuator_n,
              self.hidden_channel_n, # hidden
              self.io_channel_n], # input/output channel
              -1)

    input_ch = io[:1] if self.separated_input_output_channels else io
    body_mask = tf.cast(get_body_mask(x), tf.float32)
    valid_body_sensor = sensor * body_mask

    input_mask_bool = tf.reduce_any(tf.greater(valid_body_sensor, 0.5),
                                     axis=-1, keepdims=True)

    input_mask_idx = tf.cast(tf.where(input_mask_bool), tf.int64)

    not_input_mask = tf.cast(
        tf.expand_dims(
            tf.reduce_all(
                tf.math.logical_not(input_mask_bool), -1), -1), tf.float32)

    input_ch_shape = tf.cast(tf.shape(input_ch), tf.int64)
    sensor_inputs_mat = tf.cast(tf.scatter_nd(input_mask_idx,
                                              sensor_inputs_flat,
                                              input_ch_shape), tf.float32)

    not_sensor_inputs_mat = tf.expand_dims(tf.reduce_sum(not_input_mask*input_ch,
                                                         axis=-1), -1)

    new_input_ch = sensor_inputs_mat + not_sensor_inputs_mat

    new_io = tf.concat([new_input_ch, io[-1:]], -1) \
      if self.separated_input_output_channels else new_input_ch

    return tf.concat([energy, static, sensor, actuator, hidden, new_io], -1)

  #@tf.function
  def call(self, x, sensor_inputs=None, fixed_body_idx=None,
           body_limitation=False, fire_rate=None,
           manual_noise=None, step_size=1.0, state_update=True,
           video_making=False):
    pre_limit_mask = get_body_mask(x) if body_limitation else get_energy_mask(x)

    if sensor_inputs is not None:
      x = self.apply_sensor_inputs(x, sensor_inputs)

    dx = self.dmodel(x)*step_size
    if self.add_noise:
      if manual_noise is None:
        residual_noise = tf.random.normal(tf.shape(dx), 0., 0.02)
      else:
        residual_noise = manual_noise
      dx += residual_noise

    if fire_rate is None:
      fire_rate = self.fire_rate
    update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= fire_rate

    if self.add_control_identifier_channel:
      fixed_control_mask = tf.concat(
        [tf.fill(tf.shape(x[:, :, :, :1]), True),
         tf.fill(tf.shape(x[:, :, :, :1]), False),
         tf.fill(tf.shape(x[:, :, :, 2:]), True)], -1)
      update_mask = tf.math.logical_and(fixed_control_mask, update_mask)

    if state_update:
      newx = x + dx * tf.cast(update_mask, tf.float32)
    else:
      newx = tf.math.tanh(dx * tf.cast(update_mask, tf.float32) + dx *
                       tf.cast(tf.math.logical_not(update_mask), tf.float32))

    post_limit_mask = get_body_mask(x) if body_limitation else get_energy_mask(x)
    limit_mask = pre_limit_mask & post_limit_mask
    concat_limit_and_unlimited_mask = tf.concat(
        [tf.cast(limit_mask, tf.float32),
         tf.ones(tf.shape(limit_mask))], -1)
    complete_limit_mask = tf.repeat(concat_limit_and_unlimited_mask,
            repeats=[self.limited_channel_n,self.unlimited_channel_n],
            axis=-1)

    newx *= complete_limit_mask
    if fixed_body_idx is not None:
      concat_zeros_ones_mask = tf.concat(
          [tf.zeros(tf.shape(limit_mask[:1])),
           tf.ones(tf.shape(limit_mask[:1]))], -1)
      body_brain_mask = tf.repeat(concat_zeros_ones_mask,
              repeats=[self.body_energy_channel_n,self.brain_channel_n],
              axis=-1)

      concat_ones_and_body_brain_mask = tf.concat(
          [tf.ones(tf.shape(body_brain_mask)),
           body_brain_mask], 0)

      changing_body_mask = tf.repeat(concat_ones_and_body_brain_mask,
              repeats=[fixed_body_idx, tf.shape(limit_mask)[0]-fixed_body_idx],
              axis=0)
      fixed_body_mask = 1-changing_body_mask
      newx = newx*changing_body_mask + x*fixed_body_mask

    if sensor_inputs is not None and video_making:
      newx = self.apply_sensor_inputs(newx, sensor_inputs)

    return newx if self.clip_value == 0 else tf.clip_by_value(newx,
                                                              -self.clip_value,
                                                              self.clip_value)

  #@tf.function
  def get_actuators(self, x, act_func=None):
    energy, static, sensor, actuator, hidden, io =\
    tf.split(x,
             [self.energy_ctrl_channel_n, # energy (+control id)
              self.body_static_n,
              self.body_sensor_n,
              self.body_actuator_n,
              self.hidden_channel_n, # hidden
              self.io_channel_n], # input/output channel
              -1)

    body_mask = tf.cast(get_body_mask(x), tf.float32)
    valid_body_actuator = actuator * body_mask

    output_mask_bool = tf.reduce_any(tf.greater(valid_body_actuator, 0.5),
                                     axis=-1, keepdims=True)
    output_ch = io[-1:] if self.separated_input_output_channels else io
    output_values = output_ch[output_mask_bool]

    actuator_output = tf.reshape(output_values, [tf.shape(x)[0], -1])

    return actuator_output.numpy() \
      if act_func is None else act_func(actuator_output).numpy()

  def get_body_grid_and_fixed_body_nca(self, x):
    energy, body, hidden, io =\
    tf.split(x,
             [self.energy_ctrl_channel_n, # energy (+control id)
              self.body_channel_n,
              self.hidden_channel_n, # hidden
              self.io_channel_n], # input/output channel
              -1)

    body_mask = tf.cast(get_body_mask(x), tf.float32)
    valid_body = body * body_mask
    maximum_per_cell = tf.reduce_max(valid_body, axis=-1, keepdims=True)
    class_mask_bool = tf.equal(valid_body,maximum_per_cell)
    new_body_tf = tf.cast(class_mask_bool, tf.float32)

    body_grid = tf.argmax(new_body_tf, axis=-1).numpy()
    body_zeros = np.zeros_like(new_body_tf.numpy())

    body_flat = np.reshape(body_zeros, (-1, body_zeros.shape[-1]))
    body_flat[np.arange(body_flat.shape[0]), body_grid.flatten()] = 1.0
    new_body = np.reshape(body_flat, body_zeros.shape)

    body_mask_np = body_mask.numpy()
    new_body *= body_mask_np
    body_grid += 1
    body_grid *= np.squeeze(body_mask_np.astype(np.int64), axis=-1)

    control_id_channel = body_mask_np if self.is_body_limited_control_id \
      else np.ones_like(body_mask_np)
    updated_energy_ctrl_channel = np.concatenate((body_mask_np,
                                                  control_id_channel),
                                                 -1) \
      if self.add_control_identifier_channel else body_mask_np

    fixed_body_nca = np.concatenate((updated_energy_ctrl_channel, new_body,
                                     np.zeros_like(hidden.numpy()),
                                     np.zeros_like(io.numpy())), -1)

    return body_grid, fixed_body_nca

  def body_grid_2_fixed_body_nca(self, body_grid):
    body_grid_np = np.asarray(body_grid)
    body_grid_shape = body_grid_np.shape
    body_grid_exp = np.expand_dims(body_grid_np, -1)
    body_mask_np = (body_grid_exp != 0).astype(np.float32)
    control_id_channel = body_mask_np if self.is_body_limited_control_id \
      else np.ones_like(body_mask_np)

    updated_energy_ctrl_channel = np.concatenate((body_mask_np,
                                                  control_id_channel),
                                                 -1) \
      if self.add_control_identifier_channel else body_mask_np

    body_shape = body_grid_shape + (self.body_channel_n,)
    new_body = np.zeros(body_shape)
    for i in range(1, self.body_channel_n+1):
      new_body[:,:,:,i-1] = (body_grid_np == i).astype(np.float32)

    hidden_shape = body_grid_shape + (self.hidden_channel_n,)
    io_shape = body_grid_shape + (self.io_channel_n,)

    fixed_body_nca = np.concatenate((updated_energy_ctrl_channel, new_body,
                                     np.zeros(hidden_shape),
                                     np.zeros(io_shape)), -1)

    return fixed_body_nca

  @staticmethod
  def grid_similarity(grid):
    similarity_score_list = []
    for i in range(grid.shape[0]):
      for j in range(i+1, grid.shape[0]):
        similarity_score = grid[i] == grid[j]
        similarity_score_list.append(np.mean(similarity_score))

    return np.mean(similarity_score_list)

  #@tf.function
  def predict(self, x, sensor_inputs=None, fixed_body_idx=None,
              body_limitation=False, act_func=None, ca_steps=None):
    iter_n = ca_steps
    if iter_n is None:
      iter_n = tf.random.uniform([],
                                 self.ca_steps_per_action[0],
                                 self.ca_steps_per_action[1],
                                 tf.int32)

    for i in tf.range(iter_n):
      x = self(x, sensor_inputs=sensor_inputs, fixed_body_idx=fixed_body_idx,
               body_limitation=body_limitation)

    return x

  def build_body(self, x):
    iter_n = tf.random.uniform([],
                               self.ca_steps_build_body[0],
                               self.ca_steps_build_body[1],
                               tf.int32)
    newx = self.predict(x, ca_steps=iter_n)
    body_grid, fixed_body_nca = self.get_body_grid_and_fixed_body_nca(newx)
    return body_grid, fixed_body_nca


  def act(self, x, sensor_inputs, act_func=None):
    #print("act sensor_inputs", sensor_inputs)
    newx = self.predict(x, sensor_inputs=sensor_inputs,
                        fixed_body_idx=0,
                        body_limitation=self.act_body_limitation)

    return newx.numpy(), self.get_actuators(newx, act_func)

  def generate_channel_title(self):
    title_list = ["Body", "Control flag", "Tissue"]

    if self.body_sensor_n == 1:
      title_list.append("Sensor")
    elif self.body_sensor_n > 1:
      for s_idx in range(self.body_sensor_n):
        title_list.append("Sensor #"+str(s_idx+1))

    if self.body_actuator_n == 1:
      title_list.append("Actuator")
    elif self.body_actuator_n > 1:
      for a_idx in range(self.body_actuator_n):
        title_list.append("Actuator #"+str(a_idx+1))

    if self.hidden_channel_n == 1:
      title_list.append("Hidden")
    elif self.hidden_channel_n > 1:
      for h_idx in range(self.hidden_channel_n):
        title_list.append("Hidden #"+str(h_idx+1))

    if self.separated_input_output_channels:
      title_list.append("Input")
      title_list.append("Output")
    else:
      title_list.append("Input/Output")

    return title_list

  def get_full_nca_image(self, x, title=None, batch_idx=0, channel_names=False):
    xb = None
    if tf.is_tensor(x):
      xb = x.numpy()[batch_idx]
    else:
      xb = x[batch_idx]

    fig = plt.figure(figsize=(8,6))
    if title is not None:
      fig.suptitle(title)

    channel_name_list = []
    if channel_names:
      channel_name_list = self.generate_channel_title()

    vmax = self.clip_value if self.clip_value != 0 else 1
    sqrt_channel_n = np.sqrt(self.channel_n)
    rows = int(np.round(sqrt_channel_n))
    cols = int(np.ceil(sqrt_channel_n))
    for i in range(rows*cols):
      plt.subplot(rows,cols,i+1)
      if i < self.channel_n:
        plt.xticks([], [])
        plt.yticks([], [])
        if channel_names:
          plt.title(channel_name_list[i], fontsize=10, pad=2)
        else:
          plt.title("Channel #"+str(i+1), fontsize=10, pad=2)
        plt.imshow(xb[:,:,i], cmap="bwr", vmin=-vmax, vmax=vmax)
        #plt.pcolor(np.ones_like(xb[:,:,0]), edgecolors='k', linewidths=1, cmap="gray", vmin=0, vmax=1)
      else:
        plt.axis('off')
        plt.imshow(np.zeros_like(xb[:,:,0]), cmap="bwr", vmin=-vmax, vmax=vmax)
        #plt.pcolor(np.ones_like(xb[:,:,0]), edgecolors='k', linewidths=1, cmap="gray", vmin=0, vmax=1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(cax=cbar_ax, ticks=[-5,-4,-3,-2,-1, 0, 1,2,3,4,5])

    fig_array = utils.fig2array(fig)
    plt.close()
    return fig_array

if __name__ == "__main__":
  nca = BodyBrainNCA(hidden_channel_n=5, body_sensor_n=2)
  nca.dmodel.summary()

  size = 11

  batch_size = 2
  #x = tf.Variable(tf.random.uniform((batch_size,size,size,nca.channel_n), minval=0.0, maxval=0.7))
  x0 = tf.random.uniform((1,size,size,nca.channel_n), minval=0.0, maxval=0.7)
  x = tf.Variable(tf.repeat(x0, batch_size, axis=0))
  print("tf.shape(x)", tf.shape(x))

  body_grid, old_x2 = nca.build_body(x)
  x2 = nca.body_grid_2_fixed_body_nca(body_grid)
  print("old_x2 == x2", np.mean(old_x2 == x2))
  #number_sensors = np.sum(body_grid[0] == 2)

  number_sensors1 = np.sum(body_grid[0] == 2)
  number_sensors2 = np.sum(body_grid[0] == 3)
  number_sensors = number_sensors1 + number_sensors2

  input_values = 0.1*np.arange(batch_size*number_sensors).reshape((batch_size, number_sensors))#np.ones((batch_size, np.sum(body_grid == 2))) * 2
  print("input_values", input_values)
  print("tf.shape(x) 2", tf.shape(x))
  print("grid similarity", BodyBrainNCA.grid_similarity(body_grid))
  #newx = nca.predict(x,input_values, fixed_body_idx=0)

  newx, action = nca.act(x2,utils.reorganize_obs(body_grid[0], nca.body_sensor_n, input_values))
  newx2, action2 = nca.act(newx,utils.reorganize_obs(body_grid[0], nca.body_sensor_n, input_values/2))

  # for i in range(batch_size):
  #   #print(np.mean(np.square(newx[i,:,:,:nca.body_energy_channel_n]-x[i,:,:,:nca.body_energy_channel_n])))
  #   #print(np.mean(np.square(newx[i]-x[i])))
  #   print("batch", i)
  #   print(newx[i,:,:,:nca.body_energy_channel_n])

  #   print(body_grid[i])

  print(action)

  print(body_grid[0])
  plt.figure(1)
  img1 = nca.get_full_nca_image(old_x2, batch_idx=0)
  plt.imshow(img1)
  plt.axis('off')
  plt.tight_layout()
  # plt.show()
  plt.figure(2)
  img2 = nca.get_full_nca_image(old_x2, batch_idx=1)
  plt.imshow(img2)
  plt.axis('off')
  plt.tight_layout()
  #plt.show()
  plt.figure(3)
  img3 = nca.get_full_nca_image(x2, batch_idx=0)
  print(newx2[0,:,:,0])
  plt.imshow(img3)
  plt.axis('off')
  plt.tight_layout()
  plt.figure(4)
  img4 = nca.get_full_nca_image(x2, batch_idx=1)
  plt.imshow(img4)
  plt.axis('off')
  plt.tight_layout()
  # # plt.show()
  # plt.figure(5)
  # img5 = nca.get_full_nca_image(newx, batch_idx=1)
  # plt.imshow(img5)
  # plt.axis('off')
  # plt.tight_layout()
  # #plt.show()
  # plt.figure(6)
  # img6 = nca.get_full_nca_image(newx2, batch_idx=1)
  # plt.imshow(img6)
  # plt.axis('off')
  # plt.tight_layout()


  plt.show()
