import numpy as np
import pickle
from copy import deepcopy
import os

from utils.data_arg_parser import get_parser
from make_ctrls import *
from collectors import rollout_collector, noisy_rollout_collector
from cg_dynamics.rollout_utils import replay_rollout

CTRLS = {
    'energy': get_energy_controller(),
    'zero': get_zero_controller(),
    'noisy': get_noisy_contorller()
}

def main():
    args = get_parser()
    ctrl = CTRLS[args.nominal_ctrl]

    if args.rollout_type == 'iid':
        trajs = iid_rollouts(ctrl, args, expert=True)
    elif args.rollout_type == 'pert':
        trajs = perturb_ctrl_rollouts(ctrl, args, expert=True)

    os.makedirs(args.save_data_path, exist_ok=True)
    data_path = os.path.join(args.save_data_path, args.dataset_name)
    with open(data_path, "wb") as fp:
        pickle.dump(trajs, fp)


def iid_rollouts(ctrl, args, n_rollouts=None, expert=True, h=None):
    """Collect IID rollouts from different initial conditions."""

    if n_rollouts is None: n_rollouts = args.n_rollouts
    dt, horizon = args.dt, args.horizon

    get_rollout = rollout_collector(ctrl, args)
    trajectories = []

    while len(trajectories) < n_rollouts:
        ic = get_starting_state(args)
        num_steps, action_seq = get_rollout(ic)
        traj = replay_rollout(ic, action_seq, dt, horizon, h=h)
        
        if should_add_traj(expert, num_steps, args) is True:
            trajectories.append(traj)

    return trajectories

def perturb_ctrl_rollouts(ctrl, args, expert=True):
    """Collect rollouts by perturbing action sequences."""

    def noisy_seq(seq):
        """Add uniformly generated noise to action sequence."""

        seq = np.array(seq)
        return list(seq + np.random.uniform(size=seq.shape, low=-0.2, high=0.2))

    def replay_with_noise(init_traj, ic, args):
        """Replay trajectory with noisy action sequence."""

        orig_action_seq = init_traj['action_seq']
        pert_action_seq = noisy_seq(orig_action_seq)
        return replay_rollout(ic, pert_action_seq, args.dt, args.horizon)
    
    trajectories = []
    while len(trajectories) < args.n_rollouts * args.n_pert_rollouts:

        # collect an initial rollout
        print('Searching for initial trajectory...')
        init_traj = iid_rollouts(ctrl, args, n_rollouts=1, expert=expert)[0]
        ic = init_traj['init_state']
        trajectories.append(init_traj)
        print('...found initial trajectory.\n')

        # collect trajectories with perturbed action seqs
        pert_idx = 0
        while pert_idx < args.n_pert_rollouts:
            traj = replay_with_noise(init_traj, ic, args)
            num_steps = traj['n_steps']

            if should_add_traj(expert, num_steps, args) is True:
                print(f'Found perturbed trajectory: {pert_idx}')
                trajectories.append(traj)
                pert_idx += 1
        print('')

    return trajectories

def test_ctrls(ctrls, num_trials, args):
    """Collect rollouts for multiple controllers from the same
    initial condition.
    
    Params:
        ctrls: dictionary of the form {'name': controller_fn}
        num_trials: number of trials to perform
        args: train.py command line arguments.
    """
    
    def create_dict(steps, ic):
        return {'init_state': deepcopy(ic), 
            'success': [n > args.success_n_steps for n in steps]}
    
    results = []
    for idx in range(num_trials):
        print(f'Current index is {idx}')
        ic = get_starting_state(args)

        steps = []
        for name, ctrl in ctrls.items():
            get_rollout = rollout_collector(ctrl, args)
            n_steps, _ = get_rollout(ic)
            print(f'{name.capitalize()} ctrl walked {n_steps} steps.')
            steps.append(n_steps)
        
        results_dict = create_dict(steps, ic)
        results.append(results_dict)
        print('')

    return results

def find_success_hcbf(learned_h, safe_ctrl, baseline_ctrl, args):
    """Find a trajectory where the HCBF-based safe controller walks
    but the baseline controller does not.
    
    Params:
        learned_h: Learned function h(x) mapping states to real numbers.
        safe_ctrl: HCBF-QP-based safe controller.
        baseline_ctrl: Baseline control policy (energy-based or zero ctrl).
        args: train.py command line arguments.    
    """

    safe_collector = rollout_collector(safe_ctrl, args)
    baseline_collector = rollout_collector(baseline_ctrl, args)
    dt, horizon = args.dt, args.horizon

    while True:
        ic = get_starting_state(args)
        
        num_steps_safe, action_seq_safe = safe_collector(ic)
        print(f'Safe ctrl walked {num_steps_safe} steps.')
        num_steps_baseline, baseline_action_seq = baseline_collector(ic)
        print(f'Baseline ctrl walked {num_steps_baseline} steps.\n')

        if num_steps_safe >= 10 and num_steps_baseline <= 5:
            print('Found successful trajectory!')
            safe = replay_rollout(ic, action_seq_safe, dt, horizon,
                                    h=learned_h)
            baseline = replay_rollout(ic, baseline_action_seq, dt, 
                                        horizon)
            return safe, baseline
        

def should_add_traj(expert, num_steps, args):
    """Logic that determines whether trajectory should be added to list."""

    if expert is True and num_steps >= args.success_n_steps:
        return True
    elif expert is False:
        return True
    return False

def get_starting_state(args):
    """Create an initial state by perturbing a state on the 
    passive limit cycle."""

    passive_state = np.array([0.0, 0.0, 0.4, -2.0])    
    level = 0.02 if args.fix_left is False else 0.0

    noise = np.random.uniform(
        low=[-1 * level, -0.2, -1 * level, -0.5], 
        high=[level, 0.2, level, 0.5], 
        size=(4,)
    )
    return passive_state + noise


if __name__ == '__main__':
    main()