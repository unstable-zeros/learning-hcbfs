"""Energy based controllers for the compass gait walker.

Based on
    Limit cycles in a passive compass gait biped and passivity-mimicking control laws.
    A. Goswami, B. Espiau, and A. Keramane. 1997

"""

import numpy as np
from collections import namedtuple

import cg_dynamics.compass_gait as compass_gait

PARAMS = namedtuple('CompassGaitParams',
        ['mass_hip', 'mass_leg', 'length_leg', 'center_of_mass_leg', 'gravity', 'slope'],
        defaults=[10.0, 5.0, 1.0, 0.5, 9.81, 0.0525])
TOL = 1e-4

class EnergyBasedController(object):

    def __init__(self, energy_reference, lam, params=PARAMS()):
        self.energy_reference = energy_reference
        self.lam = lam
        self.params = params

    def get_action(self, cg_state):
        # compute the current (relative) energy of the cg_state
        current_energy = compass_gait.KineticEnergy(cg_state, self.params) + compass_gait.RelativePotentialEnergy(cg_state, self.params)
        Ediff = current_energy - self.energy_reference

        denom = cg_state[compass_gait._STANCE_VEL] - cg_state[compass_gait._SWING_VEL]
        
        # avoid singularity in denominator
        if np.abs(denom) <= TOL:
            return np.array([0.0, 0.0])

        return np.array([self.lam * Ediff / denom, 0.0])


