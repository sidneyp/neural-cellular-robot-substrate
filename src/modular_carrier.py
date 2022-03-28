"""
Modular ball chaser robot.

To play yourself, type:

python modular_ball_chaser.py

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
from modular_carrier_dynamics import ModularRobot
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

BALL_SIZE = 1

BALL_COLOR = [0.5, 0.0, 0.0]

GOAL_SIZE = 5

GOAL_COLOR = [0.1, 0.8, 0.1]

def calculatePlayfield(max_grid_size):
  return max(50, max_grid_size * 5 + 35)

class CollisionDetector(contactListener):
  def __init__(self, env):
    contactListener.__init__(self)
    self.env = env

  def BeginContact(self, contact):
    u1 = contact.fixtureA.body.userData
    u2 = contact.fixtureB.body.userData
    if u1 or u2: # Ball collision
      self.env.ball_touch = True

class Ball:
  def __init__(self, world, predefined_ball_pos, predefined_ball_pos_noise,
               ball_idx=None):
    self.world = world
    if ball_idx is None:
      ball_idx = np.random.randint(len(predefined_ball_pos))

    self.position = predefined_ball_pos[ball_idx % len(predefined_ball_pos)]
    self.position += np.random.uniform(-predefined_ball_pos_noise,
                                       predefined_ball_pos_noise, size=2)

    self.ball_body = world.CreateDynamicBody(
      position=self.position,
      angle=0,
      fixtures=[
        fixtureDef(
          shape=circleShape(radius=BALL_SIZE, pos=(0,0)),
          density=0.1,
          ),
        ],
      )
    self.ball_body.color = BALL_COLOR
    self.ball_body.userData = self.ball_body

  def step(self, dt):
    # Force
    forw = self.ball_body.GetWorldVector((0, 1))
    side = self.ball_body.GetWorldVector((1, 0))
    v = self.ball_body.linearVelocity
    vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
    vs = side[0] * v[0] + side[1] * v[1]  # side speed

    self.ball_body.ApplyForceToCenter(
      (
        -vf * forw[0] -vs * side[0],
        -vf * forw[1] -vs * side[1],
      ),
      True,
    )

  def draw(self, viewer):
    from gym.envs.classic_control import rendering
    for f in self.ball_body.fixtures:
      trans = f.body.transform
      t = rendering.Transform(translation=trans * f.shape.pos)
      viewer.draw_circle(
        f.shape.radius, 20, color=self.ball_body.color
      ).add_attr(t)

  def destroy(self):
    self.world.DestroyBody(self.ball_body)
    self.ball_body = None

class Goal:
  def __init__(self, world, predefined_goal_pos, goal_idx=None):
    self.world = world
    if goal_idx is None:
      goal_idx = np.random.randint(len(predefined_goal_pos))

    self.position = predefined_goal_pos[goal_idx % len(predefined_goal_pos)]

    GOAL_POLY = [
      (-GOAL_SIZE, +GOAL_SIZE),
      (+GOAL_SIZE, +GOAL_SIZE),
      (+GOAL_SIZE, -GOAL_SIZE),
      (-GOAL_SIZE, -GOAL_SIZE),
    ]

    self.goal_body = world.CreateStaticBody(
      position=self.position,
      angle=0,
      fixtures=[
        fixtureDef(
          shape=polygonShape(
            vertices=[(mx,my) for mx, my in GOAL_POLY]
            ),
          maskBits=0x000,
          ),
        ],
      )
    self.goal_body.color = GOAL_COLOR
    self.goal_body.userData = self.goal_body

  def draw(self, viewer):
    for f in self.goal_body.fixtures:
      trans = f.body.transform
      path = [trans * v for v in f.shape.vertices]
      viewer.draw_polygon(path, color=self.goal_body.color)

  def destroy(self):
    self.world.DestroyBody(self.goal_body)
    self.goal_body = None

class ModularCarrier(gym.Env, EzPickle):
  metadata = {
    "render.modes": ["human", "rgb_array", "state_pixels"],
    "video.frames_per_second": FPS,
  }

  def __init__(self, body_grid, done_in_ball_touch,
               predefined_ball_idx=None, zoom_manual=None,
               manual_max_body_size=None):
    EzPickle.__init__(self)
    self.seed()
    self.contactListener_keepref = CollisionDetector(self)
    self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
    self.viewer = None
    self.robot = None
    self.ball = None
    self.goal = None
    self.distance_robot_ball = None
    self.reward = 0.0
    self.ball_touch = False
    self.done_in_ball_touch = done_in_ball_touch
    self.predefined_ball_idx = predefined_ball_idx
    self.robot_failure = False
    self.zoom_manual = zoom_manual

    self.body_grid = body_grid
    self.module_number = np.sum(self.body_grid != 0)

    self.sensor1_number = np.sum(body_grid == 2)
    self.sensor2_number = np.sum(body_grid == 3)
    self.sensor_number = self.sensor1_number + self.sensor2_number
    self.observation_space = spaces.Box(
      np.array(self.sensor_number*[0]).astype(np.float32),
      np.array(self.sensor_number*[+1]).astype(np.float32),
    )
    self.observations = np.zeros(self.sensor_number)

    self.actuator_number = np.sum(np.logical_or(body_grid == 4, body_grid == 5))
    self.action_space = spaces.Box(
      np.array(self.actuator_number*[-1]).astype(np.float32),
      np.array(self.actuator_number*[+1]).astype(np.float32),
    )

    self.max_body_size = manual_max_body_size if manual_max_body_size \
      else max(np.asarray(body_grid).shape)
    self.playfield = calculatePlayfield(self.max_body_size)

    self.sensor_sensibility = self.playfield

    self.predefined_ball_pos = [[-0.6*self.playfield, 0.0],
                                [-0.3*self.playfield, 0.0],
                                [0.3*self.playfield, 0.0],
                                [0.6*self.playfield, 0.0]]


    self.predefined_goal_pos = [[-0.6*self.playfield, 0.6*self.playfield],
                                [-0.3*self.playfield, 0.6*self.playfield],
                                [0.3*self.playfield, 0.6*self.playfield],
                                [0.6*self.playfield, 0.6*self.playfield]]


    self.predefined_ball_pos_noise = 0.05*self.playfield
    self.predefined_goal_idx = np.random.randint(len(self.predefined_goal_pos))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _destroy(self):
    if self.robot:
      self.robot.destroy()
    if self.ball:
      self.ball.destroy()
    if self.goal:
      self.goal.destroy()

  def getDistanceRobotBall(self):
    # distance_robot_ball = np.inf
    # for obj in self.robot.drawlist:
    #   for f in obj.fixtures:
    #     if type(f.shape) is not circleShape:
    #       trans = f.body.transform
    #       path = [np.asarray(trans * v) for v in f.shape.vertices]
    #       for v in path:
    #         distance_vert_ball = np.linalg.norm(v - np.asarray(
    #           self.ball.ball_body.position))
    #         if distance_vert_ball < distance_robot_ball:
    #           distance_robot_ball = distance_vert_ball
    # return distance_robot_ball
    # robot_pos = self.robot.hull.position# + np.array([self.robot_init_x,
    #                                     #            self.robot_init_y])
    # return np.linalg.norm(robot_pos - self.ball.ball_body.position)

    return np.linalg.norm(self.robot.center_tracker.position - self.ball.ball_body.position)

  def normSensorDistance(self, d):
    return np.float_power((d/self.sensor_sensibility) + 1, -2)

  def reset(self):
    self._destroy()
    self.reward = 0.0
    self.t = 0.0
    self.observations = np.zeros(self.sensor_number)
    self.ball_touch = False

    try:
      if self.zoom_manual is None:
        self.robot = ModularRobot(self.world, self.body_grid,
                                  init_x=np.random.uniform(-0.6*self.playfield,
                                                            0.6*self.playfield),
                                  init_y=-0.85*self.playfield)

        self.ball = Ball(self.world, self.predefined_ball_pos,
                            self.predefined_ball_pos_noise,
                            ball_idx=self.predefined_ball_idx)
        self.goal = Goal(self.world, self.predefined_goal_pos)
      else:
        self.robot = ModularRobot(self.world, self.body_grid,
                                  init_x=0,
                                  init_y=0)

        self.ball = Ball(self.world, self.predefined_ball_pos,
                            self.predefined_ball_pos_noise,
                            ball_idx=0)
        self.goal = Goal(self.world, self.predefined_goal_pos)


      self.distance_robot_ball = self.getDistanceRobotBall()
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
      self.ball.step(1.0 / FPS)
      self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
      self.t += 1.0 / FPS

      self.distance_robot_ball = self.getDistanceRobotBall()

      # Update sensor 1 observations
      for i in range(self.sensor1_number):
        sensor_position = self.robot.sensors1[i].position
        distance_sensor_ball = np.linalg.norm(
          sensor_position-self.ball.ball_body.position)
        self.observations[i] = self.normSensorDistance(distance_sensor_ball)

      # Update sensor 2 observations
      for i in range(self.sensor2_number):
        sensor_position = self.robot.sensors2[i].position
        distance_sensor_goal = np.linalg.norm(
          sensor_position-self.goal.goal_body.position)
        self.observations[i+self.sensor1_number] = self.normSensorDistance(
          distance_sensor_goal)

      done = False
      if action is not None:  # First step without action, called from reset()
        self.reward = 0.5*self.normSensorDistance(self.distance_robot_ball) +\
          0.5*self.normSensorDistance(np.linalg.norm(
            self.ball.ball_body.position - self.goal.goal_body.position))

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

    if self.robot and self.ball:
      self.goal.draw(self.viewer)
      self.robot.draw(self.viewer)
      self.ball.draw(self.viewer)

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
  # import matplotlib.pyplot as plt
  # import datetime
  # import os

  # current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # video_dir = "carrier_" + current_time_str
  # os.makedirs(video_dir)
  # import utils
  # body_grid = np.array([[0,0,0,0,0], [0,3,2,3,0], [0,0,0,0,0]])
  # body_grid = np.array([[2,1,2], [4,1,4], [3,3,3]])
  body_grid = np.array([[2,1,0,1,2],
                        [4,3,1,3,4],
                        [0,1,1,1,0],
                        [4,1,1,1,4],
                        [1,1,1,1,1]])
  number_actions = np.sum(np.logical_or(body_grid == 4, body_grid == 5))
  a = np.array(number_actions*[0.0])

  def key_press(k, mod):
    global restart
    if k == 0xFF0D:
      restart = True
    if k == key.LEFT:
      a[0] = -1.0
      # a[2] = -1.0
    if k == key.RIGHT:
      a[0] = +1.0
      # a[2] = +1.0
    if k == key.UP:
      a[1] = +1.0
      # a[3] = +1.0
    if k == key.DOWN:
      a[1] = -1.0
      # a[3] = -1.0

  def key_release(k, mod):
    if k == key.LEFT:
      a[0] = 0
      # a[2] = 0
    if k == key.RIGHT:
      a[0] = 0
      # a[2] = 0
    if k == key.UP:
      a[1] = 0
      # a[3] = 0
    if k == key.DOWN:
      a[1] = 0
      # a[3] = 0

  env = ModularCarrier(body_grid, False, predefined_ball_idx=3)
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release

  isopen = True
  while isopen:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    # with utils.VideoWriter("modular_ball_chaser.mp4", fps=20) as vid:
    while True:
      # a = [0, 1,0,0] if steps < 4 else [1, 1,0,0]
      # a = [1, 1,0,0] if steps < 100 else [0, 0,0,0]
      isopen = env.render()
      # env_img = env.render(mode="rgb_array")
      s, r, done, info = env.step(a)
      total_reward += r
      if steps % 50 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("observation " + str(["{:+0.2f}".format(x) for x in s]))
        print("step {} reward {:+0.2f} avg_reward {:+0.2f}".format(
          steps, r, total_reward/(steps+1)))
      steps += 1

      # fname = "modular_carrier_{:04d}.png".format(steps)
      # plt.imsave(os.path.join(video_dir, fname), env_img)
      # vid.add(env_img)

      # done = steps > 200
      if done or restart or isopen == False:
      # if done or restart:
        break
    # break
  env.close()
