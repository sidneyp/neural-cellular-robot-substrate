"""
Modular light chaser robot.

To play yourself, type:

python modular_light_chaser.py

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
Modified by Sidney Pontes-Filho.
"""
import numpy as np

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import circleShape
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener

import gym
from gym import spaces
from modular_robot_dynamics import ModularRobot
from gym.utils import seeding, EzPickle

import pyglet

pyglet.options["debug_gl"] = False
from pyglet import gl

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 600
WINDOW_W = 600
WINDOW_H = 600

FPS = 50  # Frames per second

WALL_COLOR = [0.4, 0.4, 0.4]

LIGHT_SIZE = 1

LIGHT_COLOR = [0.8, 0.8, 0.0]

def calculatePlayfield(max_grid_size):
  return max(50, max_grid_size * 5 + 35)

class CollisionDetector(contactListener):
  def __init__(self, env):
    contactListener.__init__(self)
    self.env = env

  def BeginContact(self, contact):
    u1 = contact.fixtureA.body.userData
    u2 = contact.fixtureB.body.userData
    if u1 or u2: # Light collision
      self.env.light_touch = True

class Light:
  def __init__(self, world, predefined_light_pos, predefined_light_pos_noise,
               light_upper_limit, light_lower_limit,
               light_idx=None):
    self.world = world
    if light_idx is not None:
      self.position = predefined_light_pos[light_idx % len(predefined_light_pos)]
      self.position += np.random.uniform(-predefined_light_pos_noise,
                                         predefined_light_pos_noise, size=2)
    else:
      while True:
        self.position = np.random.uniform(-light_upper_limit, light_upper_limit, size=2)
        distance_origin_light = np.linalg.norm(self.position)
        if distance_origin_light > light_lower_limit:
          break

    self.bulb = world.CreateStaticBody(fixtures=[
      fixtureDef(
        shape=circleShape(radius=LIGHT_SIZE, pos=self.position),
        ),
      ],
    )
    self.bulb.color = LIGHT_COLOR
    self.bulb.userData = self.bulb

  def draw(self, viewer):
    from gym.envs.classic_control import rendering
    for f in self.bulb.fixtures:
      trans = f.body.transform
      t = rendering.Transform(translation=trans * f.shape.pos)
      viewer.draw_circle(
        f.shape.radius, 20, color=self.bulb.color
      ).add_attr(t)

  def destroy(self):
    self.world.DestroyBody(self.bulb)
    self.bulb = None

class ModularLightChaser(gym.Env, EzPickle):
  metadata = {
    "render.modes": ["human", "rgb_array", "state_pixels"],
    "video.frames_per_second": FPS,
  }

  def __init__(self, body_grid, done_in_light_touch,
               predefined_light_idx=None, zoom_manual=None,
               manual_max_body_size=None,
               with_passage=False):
    EzPickle.__init__(self)
    self.seed()
    self.contactListener_keepref = CollisionDetector(self)
    self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
    self.viewer = None
    self.robot = None
    self.light = None
    self.left_passage_wall = None
    self.right_passage_wall = None
    self.distance_robot_light = None
    self.robot_init_x = None
    self.robot_init_y = None
    self.reward = 0.0
    self.light_touch = False
    self.done_in_light_touch = done_in_light_touch
    self.predefined_light_idx = predefined_light_idx
    self.robot_failure = False
    self.with_passage = with_passage
    self.zoom_manual = zoom_manual

    self.body_grid = body_grid
    self.module_number = np.sum(self.body_grid != 0)

    self.sensor_number = np.sum(self.body_grid == 2)
    self.observation_space = spaces.Box(
      np.array(self.sensor_number*[0]).astype(np.float32),
      np.array(self.sensor_number*[+1]).astype(np.float32),
    )
    self.observations = np.zeros(self.sensor_number)

    self.actuator_number = np.sum(np.logical_or(body_grid == 3, body_grid == 4))
    self.action_space = spaces.Box(
      np.array(self.actuator_number*[-1]).astype(np.float32),
      np.array(self.actuator_number*[+1]).astype(np.float32),
    )

    self.max_body_size = manual_max_body_size if manual_max_body_size \
      else max(np.asarray(body_grid).shape)
    self.playfield = calculatePlayfield(self.max_body_size)

    self.sensor_sensibility = self.playfield

    if self.with_passage:
      self.predefined_light_pos = [[-0.6*self.playfield, 0.6*self.playfield],
                                   [-0.3*self.playfield, 0.6*self.playfield],
                                   [0.3*self.playfield, 0.6*self.playfield],
                                   [0.6*self.playfield, 0.6*self.playfield]]
    else:
      self.predefined_light_pos = [[-0.6*self.playfield, -0.6*self.playfield],
                                   [0.6*self.playfield, -0.6*self.playfield],
                                   [0.6*self.playfield, 0.6*self.playfield],
                                   [-0.6*self.playfield, 0.6*self.playfield]]

    self.predefined_light_pos_noise = 0.05*self.playfield

    self.light_upper_limit = 0.75*self.playfield
    self.light_lower_limit = 0.40*self.playfield

    self.wall_passage_roughness = 0.8*self.max_body_size

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _destroy(self):
    if self.robot:
      self.robot.destroy()
      self.light.destroy()
      if self.left_passage_wall:
        for w in self.left_passage_wall:
          self.world.DestroyBody(w)
        self.left_passage_wall = None
      if self.right_passage_wall:
        for w in self.right_passage_wall:
          self.world.DestroyBody(w)
        self.right_passage_wall = None

  def _create_passage(self):
    passage_center = np.random.uniform(-0.8*self.playfield, 0.8*self.playfield)

    left_bottom_vertices_n = np.random.randint(5,20)
    right_bottom_vertices_n = np.random.randint(5,20)


    mask_apply_roughness_left = np.ones(left_bottom_vertices_n)
    mask_apply_roughness_left[-2:] = 0

    mask_apply_roughness_right = np.ones(right_bottom_vertices_n)
    mask_apply_roughness_right[-2:] = 0

    # Left passage wall
    left_bottom_vertices_x = np.linspace(-1.1*self.playfield,
                                         passage_center-0.7*self.max_body_size,
                                         left_bottom_vertices_n) +\
      np.random.uniform(-self.wall_passage_roughness,
                        self.wall_passage_roughness,
                        size=left_bottom_vertices_n)*mask_apply_roughness_left
    left_bottom_vertices_y = np.linspace(-1.1*self.playfield,
                                         1.2*self.max_body_size,
                                         left_bottom_vertices_n) +\
      np.random.uniform(-self.wall_passage_roughness,
                        self.wall_passage_roughness,
                        size=left_bottom_vertices_n)*mask_apply_roughness_left


    left_passage_wall_vertices = list(zip(left_bottom_vertices_x,
                                  left_bottom_vertices_y))

    # Right passage wall
    right_bottom_vertices_x = np.linspace(1.1*self.playfield,
                                         passage_center+0.7*self.max_body_size,
                                         right_bottom_vertices_n) +\
      np.random.uniform(-self.wall_passage_roughness,
                        self.wall_passage_roughness,
                        size=right_bottom_vertices_n)*mask_apply_roughness_right
    right_bottom_vertices_y = np.linspace(-1.1*self.playfield,
                                         1.2*self.max_body_size,
                                         right_bottom_vertices_n) +\
      np.random.uniform(-self.wall_passage_roughness,
                        self.wall_passage_roughness,
                        size=right_bottom_vertices_n)*mask_apply_roughness_right

    right_passage_wall_vertices = list(zip(right_bottom_vertices_x,
                                  right_bottom_vertices_y))

    self.left_passage_wall = []

    for i in range(len(left_passage_wall_vertices)-1):
      vertice_part = [(-1.1*self.playfield, left_passage_wall_vertices[i][1]),
                      (-1.1*self.playfield, left_passage_wall_vertices[i+1][1]),
                      left_passage_wall_vertices[i+1],
                      left_passage_wall_vertices[i]]
      left_wall_part = self.world.CreateStaticBody(fixtures=[
        fixtureDef(
          shape=polygonShape(
            vertices=vertice_part
            ),
          density=10.0,
          ),
        ],
      )
      left_wall_part.color = WALL_COLOR
      self.left_passage_wall.append(left_wall_part)

    self.right_passage_wall = []

    for i in range(len(right_passage_wall_vertices)-1):
      vertice_part = [(1.1*self.playfield, right_passage_wall_vertices[i][1]),
                      (1.1*self.playfield, right_passage_wall_vertices[i+1][1]),
                      right_passage_wall_vertices[i+1],
                      right_passage_wall_vertices[i]]
      right_wall_part = self.world.CreateStaticBody(fixtures=[
        fixtureDef(
          shape=polygonShape(
            vertices=vertice_part
            ),
          density=10.0,
          ),
        ],
      )
      right_wall_part.color = WALL_COLOR
      self.right_passage_wall.append(right_wall_part)

  def getDistanceRobotLight(self):
    # distance_robot_light = np.inf
    # for obj in self.robot.drawlist:
    #   for f in obj.fixtures:
    #     if type(f.shape) is not circleShape:
    #       trans = f.body.transform
    #       path = [list(trans * v) for v in f.shape.vertices]
    #       for v in path:
    #         distance_vert_light = np.linalg.norm(v - self.light.position)
    #         if distance_vert_light < distance_robot_light:
    #           distance_robot_light = distance_vert_light
    # return distance_robot_light

    # robot_pos = np.array([self.robot_init_x, self.robot_init_y]) + np.asarray(self.robot.hull.position)
    # print("robot_pos", robot_pos)
    # print("self.robot_init_x", self.robot_init_x)
    # print("self.robot_init_y", self.robot_init_y)
    # print(np.array([self.robot_init_x, self.robot_init_y]))
    # print(np.asarray(self.robot.hull.position))

    return np.linalg.norm(self.robot.center_tracker.position - self.light.position)

  def normSensorDistance(self, d):
    return np.float_power((d/self.sensor_sensibility) + 1, -2)

  def reset(self):
    self._destroy()
    self.reward = 0.0
    self.t = 0.0
    self.observations = np.zeros(self.sensor_number)
    self.light_touch = False
    self.robot_init_x = np.random.uniform(-0.6*self.playfield,
                             0.6*self.playfield)
    self.robot_init_y = -0.85*self.playfield

    try:
      if self.with_passage:
        self.robot = ModularRobot(self.world, self.body_grid,
                                  init_x=self.robot_init_x,
                                  init_y=self.robot_init_y)
        # self.robot.hull.position = (self.robot_init_x, self.robot_init_y)
        self._create_passage()

      else:
        self.robot = ModularRobot(self.world, self.body_grid)
      self.light = Light(self.world, self.predefined_light_pos,
                         self.predefined_light_pos_noise,
                         self.light_upper_limit,
                         self.light_lower_limit,
                         light_idx=self.predefined_light_idx)
      self.distance_robot_light = self.getDistanceRobotLight()

    except:
      self.robot_failure = True

    return self.step(None)[0]

  def robot_failure_reward(self):
    return 0.01*(min(self.sensor_number,1)+min(self.actuator_number,2))

  def step(self, action):
    if self.robot_failure:
      done = True
      self.reward = self.robot_failure_reward()
    else:
      if action is not None:
        self.robot.gas(action)

      self.robot.step(1.0 / FPS)
      self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
      self.t += 1.0 / FPS

      self.distance_robot_light = self.getDistanceRobotLight()

      # Update sensor observations
      for i in range(self.sensor_number):
        sensor_position = self.robot.sensors[i].position
        distance_sensor_light = np.linalg.norm(
          sensor_position-self.light.position)
        self.observations[i] = np.exp(
          -distance_sensor_light/self.sensor_sensibility)
        # self.observations[i] = self.normSensorDistance(distance_sensor_light)

      done = False
      if action is not None:  # First step without action, called from reset()
        self.reward = np.exp(-self.distance_robot_light/self.sensor_sensibility)
        # self.reward = self.normSensorDistance(self.distance_robot_light)
        # x, y = self.robot.hull.position
        # if abs(x) > 1.5*self.playfield or abs(y) > 1.5*self.playfield:
        #   done = True
        #   self.reward = -1000
        if self.done_in_light_touch and self.light_touch:
          done = True
          self.reward = +1000

    return self.observations, self.reward, done, {}

  def render(self, mode="human"):
    assert mode in ["human", "state_pixels", "rgb_array"]
    if self.viewer is None:
      from gym.envs.classic_control import rendering

      self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
      self.transform = rendering.Transform()

    if "t" not in self.__dict__:
      return  # reset() not called yet

    zoom = VIDEO_W / self.playfield
    zoom *= 0.5
    if self.zoom_manual:
      zoom *= self.zoom_manual

    self.transform.set_scale(zoom, zoom)
    self.transform.set_translation(
      WINDOW_W / 2,
      WINDOW_H / 2,
    )

    if self.robot and self.light:
      self.robot.draw(self.viewer)
      self.light.draw(self.viewer)
      if self.with_passage and self.left_passage_wall and self.right_passage_wall:
        self.render_wall()

    arr = None
    win = self.viewer.window
    win.switch_to()
    win.dispatch_events()

    win.clear()
    t = self.transform
    if mode == "rgb_array":
      VP_W = VIDEO_W
      VP_H = VIDEO_H
    elif mode == "state_pixels":
      VP_W = STATE_W
      VP_H = STATE_H
    else:
      pixel_scale = 1
      if hasattr(win.context, "_nscontext"):
        pixel_scale = (
          win.context._nscontext.view().backingScaleFactor()
        )  # pylint: disable=protected-access
      VP_W = int(pixel_scale * WINDOW_W)
      VP_H = int(pixel_scale * WINDOW_H)

    gl.glViewport(0, 0, VP_W, VP_H)
    t.enable()

    self.render_playfield()
    for geom in self.viewer.onetime_geoms:
      geom.render()
    self.viewer.onetime_geoms = []
    t.disable()

    if mode == "human":
      win.flip()
      return self.viewer.isopen

    image_data = (
      pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    )
    arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
    arr = arr.reshape(VP_H, VP_W, 4)
    arr = arr[::-1, :, 0:3]

    return arr

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None

  def render_wall(self):
    for obj in self.left_passage_wall+self.right_passage_wall:
      for f in obj.fixtures:
        trans = f.body.transform
        path = [trans * v for v in f.shape.vertices]
        self.viewer.draw_polygon(path, color=obj.color)

  def render_playfield(self):
    colors = [1, 1, 1, 1.0] * 4
    polygons_ = [
      +self.playfield,
      +self.playfield,
      0,
      +self.playfield,
      -self.playfield,
      0,
      -self.playfield,
      -self.playfield,
      0,
      -self.playfield,
      +self.playfield,
      0,
    ]

    vl = pyglet.graphics.vertex_list(
      len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)
    )  # gl.GL_QUADS,
    vl.draw(gl.GL_QUADS)
    vl.delete()

if __name__ == "__main__":
  from pyglet.window import key
  # import utils
  import matplotlib.pyplot as plt
  import datetime
  import os
  import matplotlib
  matplotlib.rcParams['pdf.fonttype'] = 42
  matplotlib.rcParams['ps.fonttype'] = 42

  current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  video_dir = "light_" + current_time_str
  os.makedirs(video_dir)
  #body_grid = np.array([[0,0,1,0,0], [0,3,2,3,0], [0,0,0,0,0]])
  # body_grid = np.array([[2,1,2], [3,1,3], [1,2,1]])
  body_grid = np.array([[1,2,1,2,1],
                        [3,1,1,1,3],
                        [0,1,2,1,0],
                        [3,1,1,1,3],
                        [1,1,1,1,1]])
  number_actions = np.sum(np.logical_or(body_grid == 3, body_grid == 4))
  a = np.array(number_actions*[0.0])

  def key_press(k, mod):
    global restart
    if k == 0xFF0D:
      restart = True
    if k == key.LEFT:
      a[0] = -1.0
      a[2] = -1.0
    if k == key.RIGHT:
      a[0] = +1.0
      a[2] = +1.0
    if k == key.UP:
      a[1] = +1.0
      a[3] = +1.0
    if k == key.DOWN:
      a[1] = -1.0
      a[3] = -1.0

  def key_release(k, mod):
    if k == key.LEFT:
      a[0] = 0
      a[2] = 0
    if k == key.RIGHT:
      a[0] = 0
      a[2] = 0
    if k == key.UP:
      a[1] = 0
      a[3] = 0
    if k == key.DOWN:
      a[1] = 0
      a[3] = 0

  env = ModularLightChaser(body_grid, False, predefined_light_idx=3,
                           with_passage=False)
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release

  isopen = True
  while isopen:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    # with utils.VideoWriter("modular_light_chaser_5x5.mp4", fps=20) as vid:
    while True:
      # a = [0, 1,0,1] if steps < 10 else [1, 1,1,1]
      # a = [1, 1,1,1] if steps < 100 else [0, 0,0,0]
      isopen = env.render()
      # env_img = env.render(mode="rgb_array")
      s, r, done, info = env.step(a)
      total_reward += r
      if steps % 50 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("observation " + str(["{:+0.2f}".format(x) for x in s]))
        print("step {} reward {:+0.2f} avg_reward {:+0.2f}".format(
          steps, r, total_reward/(steps+1)))
        #print("robot pos", env.robot.hull.position)
        print("distance:", env.distance_robot_light, env.sensor_sensibility)
      steps += 1

      # env_img = env.render(mode="rgb_array")
      # fname = "modular_carrier_{:04d}.png".format(steps)
      # plt.imsave(os.path.join(video_dir, fname), env_img)
      # vid.add(env_img)

      # done = steps > 200
      if done or restart or isopen == False:
      # if done or restart:
        break
    # break
  env.close()
