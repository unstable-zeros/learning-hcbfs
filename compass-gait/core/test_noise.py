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


def test_noise(learned_h, args):
    # noise_levels = [0.1, 0.2, 0.25, 0.3, 0.4]
    noise_levels = [0.5, 0.6, 0.7]

    agent = CG_Dynamics(args.cg_params)
    cg_envir = CompassGaitEnv(dt=0.01, horizon=750, agent=agent)

    root = os.path.join(args.results_dir, 'heatmap')
    os.makedirs(root, exist_ok=True)
    fname = os.path.join(root, 'heatmap_rollouts_non_robust_2.pkl')

    no_ctrl = get_zero_controller()

    ctrls = {
        # 'energy': get_energy_controller(args.cg_params),
        # 'zero': no_ctrl,
        'non-robust': make_safe_controller(no_ctrl, learned_h, args.cg_params)
    }

    fnames = [f for f in os.listdir(root) if f.endswith('.pkl')]
    dfs = [pd.read_pickle(os.path.join(root, f)) for f in fnames]
    df = pd.concat(dfs, ignore_index=True)
    df.to_pickle(os.path.join(root, 'heatmap_rollouts_all.pkl'))

    # df.to_pickle('results/heatmap/heatmap-rollouts.pkl')
    # df = pd.read_pickle(os.path.join(root, 'heatmap_rollouts_all.pkl'))

    # df = noisy_rollouts(ctrls, cg_envir, noise_levels, fname, num_trials=200)
    plot_heatmap_noise(learned_h, df)


def noisy_rollouts(ctrls, cg_envir, noise_levels, fname, success_n_steps=5, num_trials=5):
    """Collect IID rollouts from different initial conditions."""

    cols = ['Theta_stance', 'Theta_swing', 'Vel_stance', 'Vel_swing',
                'Num_steps', 'Success', 'Noise_level', 'Controller']

    def gather_results(steps, ic, noise_level, name):
        success = int(steps) > success_n_steps
        return [*list(deepcopy(ic)), int(steps), success, noise_level, name]
   
    results = []
    for idx in range(num_trials):
        ic = get_starting_state(fix_left=True)
        print(f'Current index is {idx}\tIC is {ic}')

        for j, noise_level in enumerate(noise_levels):

            print(f'Sub index is {j}\tNoise level is {noise_level}')

            for name, ctrl in ctrls.items():
                get_rollout = rollout_collector(ctrl, cg_envir, add_noise=True,
                        noise_level=noise_level)
                n_steps, _, _ = get_rollout(ic)
                print(f'{name.capitalize()} ctrl walked {n_steps} steps.')
                results_ls = gather_results(n_steps, ic, noise_level, name)
                results.append(results_ls)

        df = pd.DataFrame(results, columns=cols)
        df.to_pickle(fname)

    return df

def plot_heatmap_noise(learned_h, df):

    df['Num_steps'] = df['Num_steps'].clip(upper=12)
    

    sns.set_style('whitegrid')
    sns.set(font_scale=0.8, font='Palatino')

    x = np.linspace(-0.3, 0.3, num=50)
    y = np.linspace(-2.6, -1.40, num=50)
    hvals1 = jax.vmap(lambda s1: jax.vmap(lambda s2: learned_h(jnp.array([0.0, s1, 0.4, s2])))(y))(x)

    def add_contour(ax, hvals, only_zero=False):
        levels = [0] if only_zero is True else 5
        cntr = ax.contour(x, y, hvals.T, levels, colors='k')
        plt.clabel(cntr, inline=1, fontsize=10)

    def add_all_contours(grid):
        print(g.axes.shape)
        safe_axes, energy_axes, zero_axes = g.axes
        [add_contour(ax, hvals1, only_zero=False) for ax in safe_axes]
        [add_contour(ax, hvals1, only_zero=True) for ax in energy_axes]
        [add_contour(ax, hvals1, only_zero=True) for ax in zero_axes]

    kwargs = {'data': df, 'x': 'Theta_swing', 'y': 'Vel_swing',
        'row': 'Controller', 'col': 'Noise_level', 'height': 2.5,
        'row_order': ['non-robust', 'energy', 'zero']}
    
    # palette = sns.color_palette("rocket_r", as_cmap=True)
    g = sns.relplot(hue='Success', **kwargs)
    # add_all_contours(g)

    g.set(xlim=(-0.4, 0.4))
    g.set(ylim=(-2.7, -1.3))
    plt.savefig('success.png')

    # palette = sns.color_palette('Spectral', 13)
    g = sns.relplot(hue='Num_steps',palette=None, **kwargs)
    # add_all_contours(g)

    g.set(xlim=(-0.4, 0.4))
    g.set(ylim=(-2.7, -1.3))
    plt.savefig('num_steps.png')

    # plt.tight_layout()

    plt.show()