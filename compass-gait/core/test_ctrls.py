import os
from copy import deepcopy
import pandas as pd

from collectors import get_starting_state


def test_ctrls_uncertain(ctrls, num_trials, args):
    """Collect rollouts for multiple controllers from the same
    initial condition.
    
    Params:
        ctrls: dictionary of the form {'name': controller_fn}
        num_trials: number of trials to perform
        args: train.py command line arguments.
    """

    root = os.path.join(args.results_dir, 'heatmap')
    os.makedirs(root, exist_ok=True)
    fname = os.path.join(root, 'heatmap_rollouts_extra_0.pkl')

    hip_masses = [6.5, 7.5, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.5, 13.5]
    cols = ['Theta_stance', 'Theta_swing', 'Vel_stance', 'Vel_swing',
                    'Num_steps', 'Success', 'Hip_mass', 'Controller']

    def gather_results(steps, ic, hip_mass, name):
        success = int(steps) > args.success_n_steps
        return [*list(deepcopy(ic)), int(steps), success, hip_mass, name]
    
    results = []
    for idx in range(num_trials):
        ic = get_starting_state(fix_left=args.fix_left)
        print(f'Current index is {idx}\tIC is {ic}')

        for j, hip_mass in enumerate(hip_masses):

            args.cg_params.mass_hip = float(hip_mass)
            print(f'Sub index is {j}\tHip mass is {args.cg_params.mass_hip}')

            for name, ctrl in ctrls.items():
                get_rollout = rollout_collector(ctrl, args)
                n_steps, _, _ = get_rollout(ic)
                print(f'{name.capitalize()} ctrl walked {n_steps} steps.')
                results_ls = gather_results(n_steps, ic, hip_mass, name)
                results.append(results_ls)
            print('')

        df = pd.DataFrame(results, columns=cols)
        df.to_pickle(fname)

    return df

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
        ic = get_starting_state(fix_left=args.fix_left)

        steps = []
        for name, ctrl in ctrls.items():
            get_rollout = rollout_collector(ctrl, args)
            n_steps, _, _ = get_rollout(ic)
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
        ic = get_starting_state(fix_left=args.fix_left)
        
        num_steps_safe, action_seq_safe = safe_collector(ic)
        print(f'Safe ctrl walked {num_steps_safe} steps.')
        num_steps_baseline, baseline_action_seq = baseline_collector(ic)
        print(f'Baseline ctrl walked {num_steps_baseline} steps.\n')

        if num_steps_safe >= 10 and num_steps_baseline <= 5:
            print('Found successful trajectory!')
            safe = replay_rollout(ic, action_seq_safe, dt, horizon,
                                    params=args.cg_params, h=learned_h)
            baseline = replay_rollout(ic, baseline_action_seq, dt, 
                                        horizon, params=args.cg_params)
            return safe, baseline