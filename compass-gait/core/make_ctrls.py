"""Each function in this file returns a controller for the 
compass gait walker.  Each controller takes as input a four-
dimensional state x and returns a two-dimensional action."""

import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp

from cg_dynamics.energy_controllers import EnergyBasedController
from cg_dynamics.dynamics import CG_Dynamics
def get_zero_controller():
    """Returns a zero controller"""

    return lambda state: np.array([0., 0.])

def get_noisy_contorller():
    """Returns a controller that itself returns random noise."""

    return lambda state: np.random.uniform(low=-0.5, high=0.5, size=(2,))

def get_energy_controller(params):
    """Returns an energy-based controller where the reference energy
    is taken from the passive limit cycle. """

    ctrl = EnergyBasedController(energy_reference=153.244, lam=0.1, params=params)
    return lambda state: ctrl.get_action(state)

def make_safe_controller(nominal_ctrl, h, params):
    """Create a safe controller using learned hybrid CBF."""

    dh = jax.grad(h, argnums=0)
    dyn = CG_Dynamics(params)

    def safe_ctrl(x):
        """Solves HCBF-QP to map an input state to a safe action u."""

        # compute action used by nominal controller
        u_nom = nominal_ctrl(x)

        # compute function values
        f_of_x, g_of_x = dyn.f(x), dyn.g(x)
        h_of_x = h(x)
        dh_of_x = dh(x)

        # setup and solve HCBF-QP with CVXPY
        u_mod = cp.Variable(len(u_nom))
        obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))
        constraints = [jnp.dot(dh_of_x, f_of_x) + u_mod.T @ jnp.dot(g_of_x.T, dh_of_x) + h_of_x >= 0]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-10)

        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return u_mod.value
        return jnp.array([0., 0.])

    return safe_ctrl