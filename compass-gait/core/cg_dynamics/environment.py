"""A compass gait implementation.

Based very heavily on drake's implementation:
    https://github.com/RobotLocomotion/drake/blob/master/examples/compass_gait/compass_gait.cc

"""

import jax.numpy as jnp
from jax import device_get
from jax import jit
import numpy as np

# enable float64
from jax.config import config
config.update('jax_enable_x64', True)

from functools import partial
import cg_dynamics.integrators as integrators

_FB_X_VEL = 7
ACTION_BOUND = 20

def _jit(*args, **kwargs):
    return jit(*args, **kwargs)

class CompassGaitEnv:

    def __init__(self, dt, horizon, agent, rng=None, switch_callback=None):
        self.dt = dt
        self.horizon = horizon
        self.rng = rng if rng is not None else np.random
        self.agent = agent
        self.lower_action_bound = [-ACTION_BOUND, -ACTION_BOUND]
        self.upper_action_bound = [ACTION_BOUND, ACTION_BOUND]
        
        # hybrid integrator expects functions with signature f(t, y)
        self.dynamics = _jit(lambda _, cg_state, action, noise: self.agent.ContinuousDynamics(cg_state, action, noise))
        self.collision_witness = _jit(lambda _, cg_state: self.agent.FootCollision(cg_state))
        self.collision_guard = _jit(lambda _, discrete_state, cg_state: self.agent.CollisionDynamics(discrete_state, cg_state))
        self.switch_callback = switch_callback
        self.reset()

    def register_switch_callback(self, switch_callback):
        self.switch_callback = switch_callback

    def remove_switch_callback(self):
        self.switch_callback = None

    def reset(self, discrete_state=None, cg_state=None):
        if discrete_state is None:
            self.discrete_state = jnp.array([0.0, 0.0])
        else:
            assert len(discrete_state) == 2
            self.discrete_state = discrete_state
        if cg_state is not None:
            assert len(cg_state) == 4
            self.cg_state = jnp.array(cg_state)
        else:
            self.cg_state = self.rng.uniform(low=-0.3, high=0.3, size=(4,))
        self.tick = 0
        return self.get_obs()

    def get_obs(self):
        return device_get(self.agent.FloatingBaseStateOut(self.discrete_state, self.cg_state))

    def cost_function(self, obs, action):
        u0, u1 = action
        return -obs[_FB_X_VEL] + 0.01 * (u0 ** 2 + u1 ** 2)

    def step(self, action, noise):

        if self.tick >= self.horizon:
            raise RuntimeError("the episode has ended-- call reset() before step")

        # clip action
        action = np.clip(action, self.lower_action_bound, self.upper_action_bound)

        # integrate the dynamics dt forward, holding action constant
        self.discrete_state, self.cg_state = integrators.hybrid_integration_v2(
                lambda t, cg_state: self.dynamics(t, cg_state, action, noise),
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

