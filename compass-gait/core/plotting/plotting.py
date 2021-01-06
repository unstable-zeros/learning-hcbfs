import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import jax
import jax.numpy as jnp
import os

def plot_phase_port(traj, args):
    """Plot the phase portrait and HCBF values for the right foot.
    
    Params:
        traj: Trajectory dictionary for compass gait walker.
        args: command line arguments for train.py.
    """

    right, h_vals, u_seq = traj['right'], traj['h_vals'], traj['u_seq']
    tolerance = 1e-7

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    v = np.where(np.logical_and(
        np.abs(u_seq[:, 0]) >= tolerance, 
        np.abs(u_seq[:, 1]) >= tolerance))
    not_v = np.where(np.logical_or(
        np.abs(u_seq[:, 0]) < tolerance, 
        np.abs(u_seq[:, 1]) < tolerance))
    v = v[0]

    # scatter ctrl actions where u_HCBF != u_nom
    ax.scatter(right[v, 0], right[v, 1], h_vals[v], color='#0800a3', 
                label=r'h(x) where $u\_{CBF} - u\_{nom} > 0$')
    
    # scatter ctrl actions where u_HCBF == u_nom
    ax.scatter(right[not_v, 0], right[not_v, 1], h_vals[not_v], 
                color='#23d1de', alpha=0.3, 
                label=r'h(x) where $u\_{CBF} - u\_{nom} = 0$')

    # plot phase portrait for right foot
    ax.plot(right[:, 0], right[:, 1], np.zeros_like(right[:, 1]), 
                label='Right foot phase portrait', color='r')
    
    matplotlib.rc('font', **{'size': 12})
    plt.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel('Right foot $\\theta$', fontsize=16)
    ax.set_ylabel('Right foot $\dot{\\theta}$', fontsize=16)
    ax.set_zlabel('h(z)', fontsize=16)

    ax.xaxis.labelpad=20
    ax.yaxis.labelpad=20
    ax.zaxis.labelpad=20

    save_name = os.path.join(args.results_dir, 'phase_portrait.png')
    plt.savefig(save_name)

def plot_heatmap(learned_h, results):
    """Plot a scatter plot of safe and unsafe initial conditions for 
    the safe ctrl and the energy-based ctrl.  Also plot heat map 
    corresponding to learned HCBF level sets."""

    sns.set_style('whitegrid')
    sns.set(font_scale=1.2)

    x = np.linspace(-0.3, 0.3, num=50)
    y = np.linspace(-2.6, -1.40, num=50)
    hvals = jax.vmap(lambda s1: jax.vmap(lambda s2: learned_h(jnp.array([0.0, s1, 0.4, s2])))(y))(x)
    
    _, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)

    points = np.vstack([results[idx]['init_state'][[1,3]] for idx in range(len(results))])
    success = np.vstack([results[idx]['success'] for idx in range(len(results))])

    c1 = np.where(success[:, 0] == True, '#8db3f0', '#f0698d')
    ax1.scatter(points[:, 0], points[:, 1], color=c1)
    # cntr_plt = ax1.contour(x, y, hvals.T, levels=[-2, -1, 0, 0.25], linewidths=3, colors='k')
    cntr_plt = ax1.contour(x, y, hvals.T, 5, linewidths=3, colors='k')
    plt.clabel(cntr_plt, inline=1, fontsize=10)
    ax1.set_title('HCBF controller')

    c2 = np.where(success[:, 1] == True, '#8db3f0', '#f0698d')
    ax2.scatter(points[:, 0], points[:, 1], color=c2)
    cntr_plt = ax2.contour(x, y, hvals.T, levels=[0], linewidths=3, colors='k')
    plt.clabel(cntr_plt, inline=1, fontsize=10)
    ax2.set_title('Energy-based controller')

    for ax in [ax1, ax2]:
        ax.set_xlabel('Swing foot $\\theta$')
        ax.set_ylabel('Swing foot $\dot{\\theta}$')

    plt.show()

def plot_heatmap_uncertain(learned_h, learned_h2, df):

    sns.set_style('whitegrid')
    sns.set(font_scale=0.6, font='Palatino')

    x = np.linspace(-0.3, 0.3, num=50)
    y = np.linspace(-2.6, -1.40, num=50)
    hvals1 = jax.vmap(lambda s1: jax.vmap(lambda s2: learned_h(jnp.array([0.0, s1, 0.4, s2])))(y))(x)
    hvals2 = jax.vmap(lambda s1: jax.vmap(lambda s2: learned_h2(jnp.array([0.0, s1, 0.4, s2])))(y))(x)

    def add_contour(ax, hvals, only_zero=False):
        levels = [0] if only_zero is True else 5
        cntr = ax.contour(x, y, hvals.T, levels, colors='k')
        plt.clabel(cntr, inline=1, fontsize=10)

    def add_all_contours(grid):
        safe_one_axes, safe_axes, energy_axes, zero_axes = g.axes
        [add_contour(ax, hvals2, only_zero=False) for ax in safe_one_axes]
        [add_contour(ax, hvals1, only_zero=False) for ax in safe_axes]
        [add_contour(ax, hvals1, only_zero=True) for ax in energy_axes]
        [add_contour(ax, hvals1, only_zero=True) for ax in zero_axes]

    kwargs = {'data': df, 'x': 'Theta_swing', 'y': 'Vel_swing',
        'row': 'Controller', 'col': 'Hip_mass', 'height': 2.5,
        'row_order': ['safe-one-expert', 'safe', 'energy', 'zero']}
    
    # palette = sns.color_palette("rocket_r", as_cmap=True)
    g = sns.relplot(hue='Success', **kwargs)
    add_all_contours(g)

    palette = sns.color_palette('Spectral', 13)
    g = sns.relplot(hue='Num_steps', palette=palette, **kwargs)
    add_all_contours(g)

    # plt.tight_layout()

    plt.show()
    