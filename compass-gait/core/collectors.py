import numpy as np

from cg_dynamics.compass_gait import CompassGaitEnv
from cg_dynamics.rollout_utils import replay_rollout, setup_env


def rollout_collector(ctrl, args):
    """Returns function that collects a rollout from a fixed IC."""

    dt, horizon = args.dt, args.horizon

    def rollout(ic):
        """Performs one rollout with given controller from a fixed IC."""

        cg_envir, is_done, total_cost, action_seq = setup_env(ic, dt, horizon)

        while not is_done:
            action = ctrl(cg_envir.cg_state)
            action_seq.append(action)
            _, cost, is_done = cg_envir.step(action)
            total_cost += cost

        num_steps = cg_envir.discrete_state[1]
        print(f'[ROLLOUT] steps: {num_steps} Cost: {total_cost}')

        return num_steps, action_seq

    return rollout


def noisy_rollout_collector(ctrl, args):
    """Returns function that collects a rollout from a fixed IC."""

    def rollout(ic):
        """Performs one rollout by perturbing a fixed controller from
        a given initial condition."""

        pert_ctrl = lambda state: ctrl(state) + np.random.uniform(low=-0.2, high=0.2)
        get_rollout = rollout_collector(pert_ctrl, args)
        return get_rollout(ic)

    return rollout


