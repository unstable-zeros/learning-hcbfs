from copy import deepcopy
import pandas as pd
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
import os

from collectors import rollout_collector, get_starting_state
from cg_dynamics.environment import CompassGaitEnv
from cg_dynamics.dynamics import CG_Dynamics
from make_ctrls import *

nums = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]


def test_mass(learned_h, args):
    df = pd.read_pickle('experiments/Jan-3-non-robust-hip-mass/heatmap/heatmap_rollouts.pkl')
    df = df[df.Controller != 'safe-one-expert']

    hip_mass = 12.5
    root = os.path.join(args.results_dir, 'heatmap')
    os.makedirs(root, exist_ok=True)
    fname = os.path.join(root, f'heatmap_rollouts_mass_{hip_mass}.pkl')

    no_ctrl = get_zero_controller()
    ctrls = {'robust': make_safe_controller(no_ctrl, learned_h, args.cg_params)}

    args.cg_params.mass_hip = hip_mass
    agent = CG_Dynamics(args.cg_params)
    cg_envir = CompassGaitEnv(dt=0.01, horizon=750, agent=agent)

    vary_mass_rollouts(ctrls, cg_envir, hip_mass, fname, 
        success_n_steps=5, num_trials=600)

    #############

    hip_mass = 13.5
    root = os.path.join(args.results_dir, 'heatmap')
    os.makedirs(root, exist_ok=True)
    fname = os.path.join(root, f'heatmap_rollouts_mass_{hip_mass}.pkl')

    no_ctrl = get_zero_controller()
    ctrls = {'robust': make_safe_controller(no_ctrl, learned_h, args.cg_params)}

    args.cg_params.mass_hip = hip_mass
    agent = CG_Dynamics(args.cg_params)
    cg_envir = CompassGaitEnv(dt=0.01, horizon=750, agent=agent)

    vary_mass_rollouts(ctrls, cg_envir, hip_mass, fname, 
        success_n_steps=5, num_trials=600)
    

    # fnames = [os.path.join(root, f) for f in os.listdir(os.path.join(root))]
    # dfs = [pd.read_pickle(f) for f in fnames]
    # dfs.append(df)
    # df = pd.concat(dfs, ignore_index=True)

    # plot_heatmap(None, df)

def plot_heatmap(learned_h, df):
    df['Num_steps'] = df['Num_steps'].clip(upper=12)

    sns.set_style('whitegrid')
    sns.set(font_scale=0.8, font='Palatino')

    kwargs = {'data': df, 'x': 'Theta_swing', 'y': 'Vel_swing',
        'row': 'Controller', 'col': 'Hip_mass', 'height': 3}
    
    palette = sns.color_palette('Spectral', 13)
    g = sns.relplot(hue='Num_steps', palette=palette, **kwargs)
    plt.show()

def vary_mass_rollouts(ctrls, cg_envir, hip_mass, fname, success_n_steps=5, num_trials=5):
    """Collect IID rollouts from different initial conditions."""

    cols = ['Theta_stance', 'Theta_swing', 'Vel_stance', 'Vel_swing',
                'Num_steps', 'Success', 'Hip_mass', 'Controller']

    def gather_results(steps, ic, hip_mass, name):
        success = int(steps) > success_n_steps
        return [*list(deepcopy(ic)), int(steps), success, hip_mass, name]
   
    results = []
    for idx in range(num_trials):
        ic = get_starting_state(fix_left=True)
        print(f'Current index is {idx}\tIC is {ic}')

        for name, ctrl in ctrls.items():
            get_rollout = rollout_collector(ctrl, cg_envir, add_noise=False, noise_level=0)
            n_steps, _, _ = get_rollout(ic)
            print(f'{name.capitalize()} ctrl walked {n_steps} steps.')
            results_ls = gather_results(n_steps, ic, hip_mass, name)
            results.append(results_ls)

        df = pd.DataFrame(results, columns=cols)
        df.to_pickle(fname)

    return df