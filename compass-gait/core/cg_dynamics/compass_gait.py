"""A compass gait implementation.

Based very heavily on drake's implementation:
    https://github.com/RobotLocomotion/drake/blob/master/examples/compass_gait/compass_gait.cc

"""

import jax.numpy as jnp
import jax.scipy as jsp

import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate

from absl import logging

from jax import device_get
from jax import lax

# enable float64
from jax.config import config
config.update('jax_enable_x64', True)

from collections import namedtuple
from functools import partial

import cg_dynamics.integrators as integrators


CompassGaitContext = namedtuple('CompassGaitContext',
        ['continuous_state',  'discrete_state', 'params'])
CompassGaitParams = namedtuple('CompassGaitParams',
        ['mass_hip', 'mass_leg', 'length_leg', 'center_of_mass_leg', 'gravity', 'slope'],
        defaults=[10.0, 5.0, 1.0, 0.5, 9.81, 0.0525])


# continuous state indices
_STANCE = 0
_SWING = 1
_STANCE_VEL = 2
_SWING_VEL = 3


# discrete state indices
_TOE_POSITION = 0
_TICKS = 1


# enable/disable jit
_DISABLE_JIT = False
if _DISABLE_JIT:
    print("WARNING: jit is disabled")


def _jit(*args, **kwargs):
    if _DISABLE_JIT:
        return args[0]
    else:
        from jax import jit
        return jit(*args, **kwargs)


@partial(_jit, static_argnums=(1,))
def FootCollision(cg_state, params):
    """Guard for foot collision."""

    collision = 2 * params.slope - cg_state[_STANCE] - cg_state[_SWING]
    return jnp.maximum(collision, cg_state[_SWING] - cg_state[_STANCE])


@partial(_jit, static_argnums=(2,))
def CollisionDynamics(discrete_state, cg_state, params):
    """Collision dynamics."""

    m = params.mass_leg
    mh = params.mass_hip
    a = params.length_leg - params.center_of_mass_leg
    b = params.center_of_mass_leg
    l = params.length_leg
    cst = jnp.cos(cg_state[_STANCE])
    csw = jnp.cos(cg_state[_SWING])
    hip_angle = cg_state[_SWING] - cg_state[_STANCE]
    c = jnp.cos(hip_angle)
    sst = jnp.sin(cg_state[_STANCE])
    ssw = jnp.sin(cg_state[_SWING])

    M_floating_base = jnp.array([
        [2 * m + mh, 0, (m * a + m * l + mh * l) * cst,  -m * b * csw],
        [0, 2 * m + mh, -(m * a + m * l + mh * l) * sst, m * b * ssw],
        [(m * a + m * l + mh * l) * cst, -(m * a + m * l + mh * l) * sst, m * a * a + (m + mh) * l * l, -m * l * b * c],
        [-m * b * csw, m * b * ssw, -m * l * b * c, m * b * b]
    ])

    J = jnp.array([
        [1, 0, l * cst,  -l * csw],
        [0, 1, -l * sst, l * ssw]
    ])

    v_pre = jnp.array([0, 0, cg_state[_STANCE_VEL], cg_state[_SWING_VEL]])

    # TODO(stephentu): consider using snp.linalg.solve for the inverses

    Minv = jnp.linalg.inv(M_floating_base)

    v_post = v_pre - Minv @ J.T @ jnp.linalg.inv(J @ Minv @ J.T) @ J @ v_pre

    next_discrete_state = jnp.array([
        discrete_state[_TOE_POSITION] - 2 * l * jnp.sin(hip_angle / 2),
        discrete_state[_TICKS] + 1
    ])

    next_cg_state = jnp.array([cg_state[_SWING], cg_state[_STANCE], v_post[3], v_post[2]])

    return next_discrete_state, next_cg_state


def _DynamicsBiasTerm(cg_state, params):

    phi = params.slope
    m = params.mass_leg
    mh = params.mass_hip
    a = params.length_leg - params.center_of_mass_leg
    b = params.center_of_mass_leg
    l = params.length_leg
    g = params.gravity
    s = jnp.sin(cg_state[_STANCE] - cg_state[_SWING])
    vst = cg_state[_STANCE_VEL]
    vsw = cg_state[_SWING_VEL]
    

    return jnp.array([
      -m * l * b * vsw * vsw * s - (mh * l + m * (a + l)) * g * jnp.sin(cg_state[_STANCE]),
      m * l * b * vst * vst * s + m * b * g * jnp.sin(cg_state[_SWING])
    ])


def _MassMatrix(cg_state, params):
    m = params.mass_leg
    mh = params.mass_hip
    a = params.length_leg - params.center_of_mass_leg
    b = params.center_of_mass_leg
    l = params.length_leg
    c = jnp.cos(cg_state[_SWING] - cg_state[_STANCE])

    return jnp.array([
        [mh * l * l + m * (l * l + a * a), -m * l * b * c],
        [-m * l * b * c, m * b * b]
    ])


@partial(_jit, static_argnums=(2,))
def ContinuousDynamics(cg_state, u, params):
    M = _MassMatrix(cg_state, params)
    bias = _DynamicsBiasTerm(cg_state, params)
    B = jnp.array([
        [-1, 1],
        [1, 0]])

    # returns x_dot
    x_dot = jnp.hstack((
        jnp.array([cg_state[_STANCE_VEL], cg_state[_SWING_VEL]]),
        jsp.linalg.solve(M, B.dot(u) - bias, sym_pos=True)
    ))

    return x_dot


@partial(_jit, static_argnums=(2,))
def AbsolutePotentialEnergy(discrete_state, cg_state, params):

    m = params.mass_leg
    mh = params.mass_hip
    a = params.length_leg - params.center_of_mass_leg
    b = params.center_of_mass_leg
    l = params.length_leg
    g = params.gravity

    cos = jnp.cos
    sin = jnp.sin

    toe = discrete_state[_TOE_POSITION]

    y_toe = -toe * sin(params.slope)
    y_hip = y_toe + l * cos(cg_state[_STANCE])

    return (m * g * (y_toe + a * cos(cg_state[_STANCE])) +
            mh * g * y_hip +
            m * g * (y_hip - b * cos(cg_state[_SWING])))


@partial(_jit, static_argnums=(1,))
def RelativePotentialEnergy(cg_state, params):
    # TODO(stephentu): avoid code duplication with AbsolutePotentialEnergy

    m = params.mass_leg
    mh = params.mass_hip
    a = params.length_leg - params.center_of_mass_leg
    b = params.center_of_mass_leg
    l = params.length_leg
    g = params.gravity

    cos = jnp.cos
    sin = jnp.sin

    toe = 0.0

    y_toe = -toe * sin(params.slope)
    y_hip = y_toe + l * cos(cg_state[_STANCE])

    return (m * g * (y_toe + a * cos(cg_state[_STANCE])) +
            mh * g * y_hip +
            m * g * (y_hip - b * cos(cg_state[_SWING])))


@partial(_jit, static_argnums=(1,))
def KineticEnergy(cg_state, params):

    m = params.mass_leg
    mh = params.mass_hip
    a = params.length_leg - params.center_of_mass_leg
    b = params.center_of_mass_leg
    l = params.length_leg
    vst = cg_state[_STANCE_VEL]
    vsw = cg_state[_SWING_VEL]

    cos = jnp.cos

    return (.5 * (mh * l * l + m * a * a) * vst * vst +
            .5 * m * (l * l * vst * vst + b * b * vsw * vsw) -
            m * l * b * vst * vsw * cos(cg_state[_SWING] - cg_state[_STANCE]))


_FB_X = 0
_FB_Y = 1
_FB_Z = 2
_FB_ROLL = 3
_FB_PITCH = 4
_FB_YAW = 5
_FB_HIP_ANGLE = 6
_FB_X_VEL = 7
_FB_Y_VEL = 8
_FB_Z_VEL = 9
_FB_ROLL_VEL = 10
_FB_PITCH_VEL = 11
_FB_YAW_VEL = 12
_FB_HIP_ANGLE_VEL = 13


@partial(_jit, static_argnums=(2,))
def FloatingBaseStateOut(discrete_state, cg_state, params):

    toe = discrete_state[_TOE_POSITION]
    left_stance = lax.convert_element_type(jnp.mod(discrete_state[_TICKS], 2), jnp.int64) == 0

    left = jnp.where(left_stance, cg_state[_STANCE], cg_state[_SWING])
    right = jnp.where(left_stance, cg_state[_SWING], cg_state[_STANCE])
    leftdot = jnp.where(left_stance, cg_state[_STANCE_VEL], cg_state[_SWING_VEL])
    rightdot = jnp.where(left_stance, cg_state[_SWING_VEL], cg_state[_STANCE_VEL])

    cos = jnp.cos
    sin = jnp.sin

    return jnp.array([

        # x, y, z.
        toe * cos(params.slope) + params.length_leg * sin(cg_state[_STANCE]),
        0.0,
        -toe * sin(params.slope) + params.length_leg * cos(cg_state[_STANCE]),

        # roll, pitch, yaw.
        0.0,
        left, # Left leg is attached to the floating base.
        0.0,

        # Hip angle (right angle - left_angle).
        right - left,

        # x, y, z derivatives.
        cg_state[_STANCE_VEL] * params.length_leg * cos(cg_state[_STANCE]),
        0.0,
        -cg_state[_STANCE_VEL] * params.length_leg * sin(cg_state[_STANCE]),

        # roll, pitch, yaw derivatives.
        0.0,
        leftdot,
        0.0,

        # Hip angle derivative.
        rightdot - leftdot,

    ])

class Dynamics:

    def __init__(self):
        self._params = CompassGaitParams()

    def alpha(self, x):
        return x

    def f(self, x):
        M = _MassMatrix(x, self._params)
        b = _DynamicsBiasTerm(x, self._params)
        return jnp.hstack((x[2:], -jsp.linalg.solve(M, b, sym_pos=True)))
    
    def g(self, x):
        M = _MassMatrix(x, self._params)
        B = jnp.array([
            [-1.0, 1.0],
            [1.0, 0.0]
        ])
        return jnp.vstack((np.zeros((2, 2)), jsp.linalg.solve(M, B, sym_pos=True)))


# Environment for black box optimization

class CompassGaitEnv(object):

    def __init__(self, dt, horizon, rng=None, switch_callback=None):
        self.dt = dt
        self.horizon = horizon
        self.rng = rng if rng is not None else np.random
        self.params = CompassGaitParams()
        BOUND = 20
        self.lower_action_bound = [-BOUND, -BOUND]
        self.upper_action_bound = [BOUND, BOUND]
        
        # hybrid integrator expects functions with signature f(t, y)
        self.dynamics = _jit(lambda _, cg_state, action: ContinuousDynamics(cg_state, action, self.params))
        self.collision_witness = _jit(lambda _, cg_state: FootCollision(cg_state, self.params))
        self.collision_guard = _jit(lambda _, discrete_state, cg_state: CollisionDynamics(discrete_state, cg_state, self.params))
        self.switch_callback = switch_callback
        self.reset()

    def register_switch_callback(self, switch_callback):
        self.switch_callback = switch_callback

    def remove_switch_callback(self):
        self.switch_callback = None

    def reset(self, discrete_state=None, cg_state=None):
        if discrete_state is None:
            self.discrete_state = np.array([0.0, 0.0])
        else:
            assert len(discrete_state) == 2
            self.discrete_state = discrete_state
        if cg_state is not None:
            assert len(cg_state) == 4
            self.cg_state = np.array(cg_state)
        else:
            self.cg_state = self.rng.uniform(low=-0.3, high=0.3, size=(4,))
        self.tick = 0
        return self.get_obs()

    def get_obs(self):
        return device_get(FloatingBaseStateOut(self.discrete_state, self.cg_state, self.params))

    def cost_function(self, obs, action):
        u0, u1 = action
        return -obs[_FB_X_VEL] + 0.01 * (u0 ** 2 + u1 ** 2)

    def step(self, action):

        if self.tick >= self.horizon:
            raise RuntimeError("the episode has ended-- call reset() before step")

        # clip action
        action = np.clip(action, self.lower_action_bound, self.upper_action_bound)

        logging.debug("collision_witness before integration: %f", self.collision_witness(0.0,  self.cg_state))

        # integrate the dynamics dt forward, holding action constant
        self.discrete_state, self.cg_state = integrators.hybrid_integration_v2(
                lambda t, cg_state: self.dynamics(t, cg_state, action),
                self.collision_witness,
                integrators.DIR_POSITIVE_TO_NEGATIVE,
                self.collision_guard,
                self.discrete_state,
                self.cg_state,
                self.tick * self.dt,
                (self.tick + 1) * self.dt,
                self.switch_callback)

        cost = self.cost_function(self.get_obs(), action)

        self.tick += 1

        return self.get_obs(), cost, (self.tick == self.horizon)

