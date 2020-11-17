import numpy as np

from cg_dynamics.compass_gait import CompassGaitEnv

def setup_env(ic, dt, horizon):
    """Setup environment and initialize variables for simulation."""

    cg_envir = CompassGaitEnv(dt=dt, horizon=horizon)
    cg_envir.reset(cg_state=ic)
    is_done, total_cost, action_seq = False, 0.0, []

    return cg_envir, is_done, total_cost, action_seq

def replay_rollout(init_state, action_seq, dt, horizon, h=None):
    """Replay a trajectory to obtain the discrete state transitions.
    
    Params:
        init_state: initial state for compass gait walker.
        action_seq: sequence of actions used in initial rollout.
        
    Returns: 
        dictionary with the following data:
            x_cts: sequence of encountered continuous states.
            u_seq: sequence of actions used by controller.
            x_dis_minus: discrete states before point of jump.
            x_dis_plus: discrete state after point of jump.
            init_state: initial state used.
    """

    cg_envir = CompassGaitEnv(dt=dt, horizon=horizon)
    cg_envir.reset(cg_state=init_state)

    # register a switch callback that records the discrete state transitions
    callback = SwitchCallback()
    cg_envir.register_switch_callback(callback)

    is_done = False
    step_index, total_cost = 0, 0.0
    cts_state_seq, replay_action_seq, left_seq, right_seq, all_obs, true_dis = ([] for _ in range(6))
    if h is not None: h_vals = []

    while not is_done:
        if step_index < len(action_seq):
            action = action_seq[step_index]
        else:
            action = np.array([0., 0.])

        obs, cost, is_done = cg_envir.step(action)
        total_cost += cost
        all_obs.append(obs)

        # save countinuous states and actions used
        cts_state_seq.append(cg_envir.cg_state)
        replay_action_seq.append(action)
        true_dis.append(cg_envir.discrete_state)

        # find states of left and right feet
        if cg_envir.discrete_state[1] % 2 == 0:
            left_seq.append(cg_envir.cg_state[[0, 2]])
            right_seq.append(cg_envir.cg_state[[1, 3]])

        else:
            left_seq.append(cg_envir.cg_state[[1, 3]])
            right_seq.append(cg_envir.cg_state[[0, 2]])

        if h is not None:
            h_vals.append(h(cg_envir.cg_state))

        # update index
        step_index += 1

    print(f'[REPLAY] steps: {cg_envir.discrete_state[1]} Cost: {total_cost}')

    # convert lists to numpy arrays
    cts_state_seq = np.array(cts_state_seq)
    replay_action_seq = np.array(replay_action_seq)

    # extract points of jump
    cg_envir.remove_switch_callback()
    dis_state_minus = np.array([e[0] for e in callback.discrete_transitions])
    dis_state_plus = np.array([e[1] for e in callback.discrete_transitions])

    return_dict = {
        'x_cts': cts_state_seq, 
        'u_seq': replay_action_seq, 
        'x_dis_minus': dis_state_minus, 
        'x_dis_plus': dis_state_plus,
        'init_state': init_state,
        'left': np.array(left_seq),
        'right': np.array(right_seq),
        'obs': all_obs,
        'true_dis_state': np.array(true_dis),
        'n_steps': cg_envir.discrete_state[1],
        'total_cost': total_cost,
        'action_seq': action_seq
    }
    if h is not None:
        return_dict['h_vals'] = np.array(h_vals)

    return return_dict

class SwitchCallback(object):
    def __init__(self):
        self.discrete_transitions = []

    def __call__(self, tcur, discrete_state_before, cg_state_before,
            discrete_state_after, cg_state_after):
        del tcur
        del discrete_state_before
        del discrete_state_after
        self.discrete_transitions.append((cg_state_before, cg_state_after))
