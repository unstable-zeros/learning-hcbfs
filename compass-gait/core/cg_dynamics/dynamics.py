import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, jit
from functools import partial

class CG_Dynamics:

    def __init__(self, params):
        self.params = params

    def alpha(self, x):
        return x

    def f(self, cg_state):
        _, _, vel_st, vel_sw = cg_state
        M = self._MassMatrix(cg_state)
        bias = self._DynamicsBiasTerm(cg_state)

        return jnp.hstack((
            jnp.array([vel_st, vel_sw]),
            -jsp.linalg.solve(M, bias, sym_pos=True)
        ))

    def g(self, cg_state):
        M = self._MassMatrix(cg_state)
        B = self._InputMatrix()

        return jnp.vstack((
            jnp.zeros((2, 2)), 
            jsp.linalg.solve(M, B, sym_pos=True)
        ))

    @partial(jit, static_argnums=0)
    def FootCollision(self, cg_state):
        """Guard for foot collision."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, _, _ = cg_state
        collision = 2 * phi - theta_st - theta_sw
        return jnp.maximum(collision, theta_sw - theta_st)

    @partial(jit, static_argnums=0)
    def CollisionDynamics(self, discrete_state, cg_state):
        """Collision dynamics."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, vel_st, vel_sw = cg_state
        toe_pos, ticks = discrete_state
        
        M_floating_base = self._M_fb_matrix(cg_state)
        J = self._J_fb_matrix(cg_state)

        v_pre = jnp.array([0, 0, vel_st, vel_sw])
        Minv = jnp.linalg.inv(M_floating_base)
        v_post = v_pre - Minv @ J.T @ jnp.linalg.inv(J @ Minv @ J.T) @ J @ v_pre

        next_discrete_state = jnp.array([toe_pos - 2 * l * jnp.sin((theta_sw - theta_st) / 2), ticks + 1])
        next_cg_state = jnp.array([theta_sw, theta_st, v_post[3], v_post[2]])

        return next_discrete_state, next_cg_state

    def _M_fb_matrix(self, cg_state):
        """Mass matrix for floating base state used in collision dynamics."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, _, _ = cg_state
        cst, csw = jnp.cos(theta_st), jnp.cos(theta_sw)
        sst, ssw = jnp.sin(theta_st), jnp.sin(theta_sw)
        c = jnp.cos(theta_sw - theta_st)

        return jnp.array([
            [2 * m + mh, 0, (m * a + m * l + mh * l) * cst,  -m * b * csw],
            [0, 2 * m + mh, -(m * a + m * l + mh * l) * sst, m * b * ssw],
            [(m * a + m * l + mh * l) * cst, -(m * a + m * l + mh * l) * sst, m * a * a + (m + mh) * l * l, -m * l * b * c],
            [-m * b * csw, m * b * ssw, -m * l * b * c, m * b * b]
        ])

    def _J_fb_matrix(self, cg_state):
        """Matrix J from discrete collision dynamics."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, _, _ = cg_state

        return jnp.array([
            [1, 0, l * jnp.cos(theta_st),  -l * jnp.cos(theta_sw)],
            [0, 1, -l * jnp.sin(theta_st), l * jnp.sin(theta_sw)]
        ])

    def _DynamicsBiasTerm(self, cg_state):
        """Bias vector for continuous time dynamics."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, vel_st, vel_sw = cg_state
        s = jnp.sin(theta_st - theta_sw)

        return jnp.array([
            -m * l * b * (vel_sw ** 2) * s - (mh * l + m * (a + l)) * g * jnp.sin(theta_st),
            m * l * b * (vel_st ** 2) * s + m * b * g * jnp.sin(theta_sw)
        ])

    def _MassMatrix(self, cg_state):
        """Mass matrix for continuous time dynamics."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, _, _ = cg_state
        c = jnp.cos(theta_sw - theta_st)

        return jnp.array([
            [mh * l * l + m * (l * l + a * a), -m * l * b * c],
            [-m * l * b * c, m * b * b]
        ])

    @partial(jit, static_argnums=0)
    def ContinuousDynamics(self, cg_state, u, noise):
        """Continous time dynamics."""

        _, _, vel_st, vel_sw = cg_state
        M = self._MassMatrix(cg_state)
        bias = self._DynamicsBiasTerm(cg_state)
        B = self._InputMatrix()
        
        return jnp.hstack((
            jnp.array([vel_st, vel_sw]),
            jsp.linalg.solve(M, B.dot(u) - bias, sym_pos=True)
        )) + noise

    @staticmethod
    def _InputMatrix():
        """Input matrix B for continuous dynamics."""

        return jnp.array([
            [-1, 1],
            [1, 0]
        ])

    @partial(jit, static_argnums=0)
    def RelativePotentialEnergy(self, cg_state):
        """Relative potential energy for energy-based controller."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, _, _ = cg_state

        toe = 0.0

        y_toe = -toe * jnp.sin(phi)
        y_hip = y_toe + l * jnp.cos(theta_st)

        return (m * g * (y_toe + a * jnp.cos(theta_st)) +
                mh * g * y_hip +
                m * g * (y_hip - b * jnp.cos(theta_sw)))

    @partial(jit, static_argnums=0)
    def KineticEnergy(self, cg_state):
        """Kinetic energy for energy-based controller."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, vel_st, vel_sw = cg_state

        return (.5 * (mh * l * l + m * a * a) * (vel_st ** 2) +
                .5 * m * (l * l * (vel_st ** 2) + b * b * (vel_sw ** 2)) -
                m * l * b * (vel_sw ** 2) * jnp.cos(theta_sw - theta_st))

    def parse_params(self):
        """Returns parameters for compass gait walker."""

        m = self.params.mass_leg
        mh = self.params.mass_hip
        a = self.params.length_leg - self.params.center_of_mass_leg
        b = self.params.center_of_mass_leg
        l = self.params.length_leg
        phi = self.params.slope
        g = self.params.gravity

        return m, mh, a, b, l, phi, g

    @partial(jit, static_argnums=0)
    def FloatingBaseStateOut(self, discrete_state, cg_state):
        """Get floating base state."""

        m, mh, a, b, l, phi, g = self.parse_params()
        theta_st, theta_sw, vel_st, vel_sw = cg_state
        toe_pos, ticks = discrete_state

        left_st = lax.convert_element_type(jnp.mod(ticks, 2), jnp.int64) == 0
        left = jnp.where(left_st, theta_st, theta_sw)
        right = jnp.where(left_st, theta_sw, theta_st)
        leftdot = jnp.where(left_st, vel_st, vel_sw)
        rightdot = jnp.where(left_st, vel_sw, vel_st)

        return jnp.array([
            toe_pos * jnp.cos(phi) + l * jnp.sin(theta_st), 0.0, -toe_pos * jnp.sin(phi) + l * jnp.cos(theta_st),       # x, y, z.
            0.0, left, 0.0,     # roll, pitch, yaw. # Left leg is attached to the floating base.
            right - left,       # Hip angle (right angle - left_angle).
            vel_st * l * jnp.cos(theta_st), 0.0, -vel_st * l * jnp.sin(theta_st),                   # x, y, z derivatives.
            0.0, leftdot, 0.0,                  # roll, pitch, yaw derivatives.
            rightdot - leftdot,     # Hip angle derivative.
        ])