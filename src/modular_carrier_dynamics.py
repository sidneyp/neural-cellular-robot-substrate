"""
Top-down modular robot simulation.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
Modified by Sidney Pontes-Filho.
"""

import numpy as np
import math
from Box2D.b2 import (
  circleShape,
  fixtureDef,
  polygonShape,
  revoluteJointDef,
)


MODULE_SIZE = 1
WHEEL_SIZE = 1
SENSOR_SIZE = 0.4
MODULE_POLY = [
  (-MODULE_SIZE, +MODULE_SIZE),
  (+MODULE_SIZE, +MODULE_SIZE),
  (+MODULE_SIZE, -MODULE_SIZE),
  (-MODULE_SIZE, -MODULE_SIZE),
]

WHEEL_POLY = [
  (-WHEEL_SIZE, +WHEEL_SIZE),
  (+WHEEL_SIZE, +WHEEL_SIZE),
  (+WHEEL_SIZE, -WHEEL_SIZE),
  (-WHEEL_SIZE, -WHEEL_SIZE),
]

SENSOR_POLY = [
  (-SENSOR_SIZE, +SENSOR_SIZE),
  (+SENSOR_SIZE, +SENSOR_SIZE),
  (+SENSOR_SIZE, -SENSOR_SIZE),
  (-SENSOR_SIZE, -SENSOR_SIZE),
]

HULL_COLOR = (0.1, 0.5, 1.0)
SENSOR1_COLOR = (1.0, 0.0, 0.0)
SENSOR2_COLOR = (0.0, 1.0, 0.0)
WHEEL_COLOR = (0.0, 0.0, 0.0)
WHEEL_WHITE = (0.3, 0.3, 0.3)

ENGINE_POWER = 1000
FRICTION = 20

class ModularRobot:
  def __init__(self, world, body_grid, init_x=0, init_y=0, init_angle=0):
    self.world = world
    body_shape = body_grid.shape
    assert len(body_shape) == 2, "body_grid must be a 2D Numpy array."
    self.body_grid = body_grid
    hull_mask = np.logical_or(np.logical_or(body_grid == 1, body_grid == 2),
                              body_grid == 3)
    wheel_vertical_mask = body_grid == 4
    wheel_horizontal_mask = body_grid == 5
    wheel_mask = np.logical_or(wheel_vertical_mask, wheel_horizontal_mask)
    sensor1_mask = body_grid == 2
    sensor2_mask = body_grid == 3
    assert (np.sum(sensor1_mask) > 0 and np.sum(sensor2_mask) > 0 and
            np.sum(wheel_mask) >=2)

    # Calculate position of robot's modules
    module_pos = np.zeros(body_shape + (2,))
    module_pos_x = np.linspace(-(body_shape[0]-1),
                               +(body_shape[0]-1),
                               body_shape[0])
    module_pos_y = np.linspace(-(body_shape[1]-1),
                               +(body_shape[1]-1),
                               body_shape[1])
    for i in range(body_shape[0]):
      for j in range(body_shape[1]):
        module_pos[i,j] = [module_pos_y[j], module_pos_x[-i-1]]
        #module_pos[i,j] = [module_pos_y[j], module_pos_x[i]]

    # Hull
    hull_module_pos = module_pos[hull_mask]
    if len(hull_module_pos) == 0:
      print("No hull module!")
    hull_fixtures = []
    for x, y in hull_module_pos:
      hull_fixtures.append(fixtureDef(
        shape=polygonShape(
          vertices=[(x+mx+init_x, y+my+init_y) for mx, my in MODULE_POLY]
          ),
        density=1.0,
        )
      )

    self.hull = self.world.CreateDynamicBody(
    position=(0, 0),
    angle=0,
    fixtures=hull_fixtures
    )
    self.hull.color = HULL_COLOR

    # Wheels
    wheel_horizontal_pos = module_pos[wheel_horizontal_mask]
    wheel_pos = module_pos[wheel_mask]
    if len(wheel_pos) == 0:
      print("No wheels!")

    self.wheels = []
    for x,y in wheel_pos:
      w = self.world.CreateDynamicBody(
        position=(x+init_x, y+init_y),
        angle=0.5*np.pi if np.any(np.all(wheel_horizontal_pos == [x,y], axis=1)) else 0,
        fixtures=[
          fixtureDef(
            shape=polygonShape(
              vertices=[(mx, my) for mx, my in WHEEL_POLY]
            ),
            density=0.1,
          ),
        ],
      )
      w.color = WHEEL_COLOR
      w.wheel_rad = WHEEL_SIZE#MODULE_SIZE
      w.gas = 0.0
      w.phase = 0.0  # wheel angle
      w.omega = 0.0  # angular velocity
      rjd = revoluteJointDef(
        bodyA=self.hull,
        bodyB=w,
        localAnchorA=(x+init_x, y+init_y),
        localAnchorB=(0, 0),
        enableMotor=True,
        enableLimit=True,
        maxMotorTorque=180 * 900,
        motorSpeed=0,
        lowerAngle=0.0,
        upperAngle=0.0,
      )
      w.joint = self.world.CreateJoint(rjd)
      self.wheels.append(w)

    # Sensor 1
    sensor1_pos = module_pos[sensor1_mask]
    self.sensors1 = []
    for x,y in sensor1_pos:
      sensor = self.world.CreateDynamicBody(
        position=(x+init_x, y+init_y),
        angle=0,
        fixtures=[
          fixtureDef(
            shape=circleShape(radius=SENSOR_SIZE, pos=(0, 0)),
            density=0.001,
          ),
        ],
      )

      sensor.color = SENSOR1_COLOR
      rjd = revoluteJointDef(
        bodyA=self.hull,
        bodyB=sensor,
        localAnchorA=(x+init_x, y+init_y),
        localAnchorB=(0, 0),
        enableMotor=False,
        enableLimit=False
      )
      sensor.joint = self.world.CreateJoint(rjd)
      self.sensors1.append(sensor)

    # Sensor 2
    sensor2_pos = module_pos[sensor2_mask]
    self.sensors2 = []
    for x,y in sensor2_pos:
      sensor = self.world.CreateDynamicBody(
        position=(x+init_x, y+init_y),
        angle=0,
        fixtures=[
          fixtureDef(
            shape=circleShape(radius=SENSOR_SIZE, pos=(0, 0)),
            density=0.001,
          ),
        ],
      )

      sensor.color = SENSOR2_COLOR
      rjd = revoluteJointDef(
        bodyA=self.hull,
        bodyB=sensor,
        localAnchorA=(x+init_x, y+init_y),
        localAnchorB=(0, 0),
        enableMotor=False,
        enableLimit=False
      )
      sensor.joint = self.world.CreateJoint(rjd)
      self.sensors2.append(sensor)

    # Center tracker
    self.center_tracker = self.world.CreateDynamicBody(
      position=(init_x, init_y),
      angle=0,
      fixtures=[
        fixtureDef(
          shape=circleShape(radius=0.001, pos=(0, 0)),
          density=0.001,
          maskBits=0x000,
        ),
      ],
    )

    rjd = revoluteJointDef(
      bodyA=self.hull,
      bodyB=self.center_tracker,
      localAnchorA=(init_x, init_y),
      localAnchorB=(0, 0),
      enableMotor=False,
      enableLimit=False
    )
    self.center_tracker.joint = self.world.CreateJoint(rjd)

    self.drawlist = self.wheels + [self.hull] + self.sensors1 + self.sensors2

  def gas(self, gas):
    """control: rear wheel drive

    Args:
      gas (array): How much gas gets applied. Gets clipped between -1 and 1.
    """
    gas = np.clip(gas, -1, 1)
    for i in range(len(self.wheels)):
      self.wheels[i].gas = gas[i]

  def step(self, dt):
    for w in self.wheels:
      # Force
      forw = w.GetWorldVector((0, 1))
      side = w.GetWorldVector((1, 0))
      v = w.linearVelocity
      vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
      vs = side[0] * v[0] + side[1] * v[1]  # side speed

      w.omega += (
        dt
        * ENGINE_POWER
        * w.gas
        / (abs(w.omega) + 5.0)
      )

      w.phase += w.omega * dt

      vr = w.omega * w.wheel_rad  # rotating wheel speed
      f_force = vr  # force direction is direction of speed difference
      p_force = -vs

      # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
      f_force *= 10
      p_force *= 100

      w.omega -= dt * f_force * w.wheel_rad

      gas_force = w.gas * ENGINE_POWER - vf * FRICTION

      w.ApplyForceToCenter(
        (
          gas_force * forw[0] + p_force * side[0],
          gas_force * forw[1] + p_force * side[1],
        ),
        True,
      )

  def draw(self, viewer):
    from gym.envs.classic_control import rendering

    for obj in self.drawlist:
      for f in obj.fixtures:
        trans = f.body.transform
        if type(f.shape) is circleShape:
            t = rendering.Transform(translation=trans * f.shape.pos)
            viewer.draw_circle(
                f.shape.radius, 20, color=obj.color
            ).add_attr(t)
        else:
          path = [trans * v for v in f.shape.vertices]
          viewer.draw_polygon(path, color=obj.color)
          if "phase" not in obj.__dict__:
            continue
          a1 = obj.phase
          a2 = obj.phase + 1.2  # radians
          s1 = math.sin(a1)
          s2 = math.sin(a2)
          c1 = math.cos(a1)
          c2 = math.cos(a2)
          if s1 > 0 and s2 > 0:
            continue
          if s1 > 0:
            c1 = np.sign(c1)
          if s2 > 0:
            c2 = np.sign(c2)
          white_poly = [
            (-WHEEL_SIZE, +WHEEL_SIZE * c1),
            (+WHEEL_SIZE, +WHEEL_SIZE * c1),
            (+WHEEL_SIZE, +WHEEL_SIZE * c2),
            (-WHEEL_SIZE, +WHEEL_SIZE * c2),
          ]
          viewer.draw_polygon([trans * v for v in white_poly], color=WHEEL_WHITE)

  def destroy(self):
    self.world.DestroyBody(self.hull)
    self.hull = None
    for w in self.wheels:
      self.world.DestroyBody(w)
    self.wheels = []
    for s in self.sensors1:
      self.world.DestroyBody(s)
    self.sensors1 = []
    for s in self.sensors2:
      self.world.DestroyBody(s)
    self.sensors2 = []
