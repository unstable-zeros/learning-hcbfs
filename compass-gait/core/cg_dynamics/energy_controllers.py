"""Energy based controllers for the compass gait walker.

Based on
    Limit cycles in a passive compass gait biped and passivity-mimicking control laws.
    A. Goswami, B. Espiau, and A. Keramane. 1997

"""

import numpy as np
from collections import namedtuple

from cg_dynamics.dynamics import CG_Dynamics

TOLERANCE = 1e-4

class EnergyBasedController:

    def __init__(self, energy_reference, lam, params):
        self.energy_reference = energy_reference
        self.lam = lam
        self.params = params
        self.agent = CG_Dynamics(params)

    def get_action(self, cg_state):
        # compute the current (relative) energy of the cg_state
        current_energy = self.agent.KineticEnergy(cg_state) + self.agent.RelativePotentialEnergy(cg_state)
        Ediff = current_energy - self.energy_reference
        _, _, vel_st, vel_sw = cg_state
        denom = vel_st - vel_sw
        
        # avoid singularity in denominator
        if np.abs(denom) <= TOLERANCE:
            return np.array([0.0, 0.0])

        return np.array([self.lam * Ediff / denom, 0.0])


