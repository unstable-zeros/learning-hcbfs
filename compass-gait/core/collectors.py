import numpy as np
from numpy import random

from cg_dynamics.rollout_utils import replay_rollout

def rollout_collector(ctrl, cg_envir, add_noise, noise_level):
    """Returns function that collects a rollout from a fixed IC."""

    def rollout(ic):
        """Performs one rollout with given controller from a fixed IC."""

        cg_envir.reset(cg_state=ic)
        is_done, total_cost  = False, 0.0
        action_seq, noise_seq = [], []

        while not is_done:
            action = ctrl(cg_envir.cg_state)
            action_seq.append(action)

            if add_noise is True:
                noise = np.random.uniform(low=-noise_level, high=noise_level, size=(4,))
            else:
                noise = np.zeros(shape=(4,))
            noise_seq.append(noise)
            
            _, cost, is_done = cg_envir.step(action, noise)
            total_cost += cost

        num_steps = cg_envir.discrete_state[1]
        print(f'[ROLLOUT] steps: {num_steps} Cost: {total_cost}')

        return num_steps, action_seq, noise_seq

    return rollout

def get_starting_state(fix_left=False):
    """Create an initial state by perturbing a state on the 
    passive limit cycle."""

    passive_state = np.array([0.0, 0.0, 0.4, -2.0])    
    level = 0.02 if fix_left is False else 0.0

    noise = np.random.uniform(
        low=[-level, -0.2, -level, -0.5], 
        high=[level, 0.2, level, 0.5], 
        size=(4,)
    )
    return passive_state + noise