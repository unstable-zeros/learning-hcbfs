import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import itertools
from sklearn.neighbors import KDTree

N_CHUNKS = 14
N_COLS, N_ROWS = 5, 3

def get_chunk(row, bins):
    return np.digitize(row['Vel_swing'], bins)

def get_df_data(df):
    def extract(df):
        left_theta = df['Theta_stance'].to_numpy()
        right_theta = df['Theta_swing'].to_numpy()
        left_vel = df['Vel_stance'].to_numpy()
        return [left_theta, right_theta, left_vel]

    in_df = df.loc[df['Outlier'] == False]
    out_df = df.loc[df['Outlier'] == True]
    
    return extract(in_df), extract(out_df)

def kdtree_load_data(args):
    with open(args.train_data_path, 'rb') as fp:
        data = pickle.load(fp)

    if args.n_train_rollouts > len(data):
        print(f'[INFO] You requested {args.n_train_rollouts} rollouts, but the dataset only contains {len(data)} rollouts.')
        print(f'[INFO] Defaulting to {len(data)} rollouts.')
        args.n_train_rollouts = len(data)
 
    loaded_data = {
        'x_cts': np.vstack([x['x_cts'] for x in data[:args.n_train_rollouts]]),
        'x_dis_minus': np.vstack([x['x_dis_minus'] for x in data]),
        'x_dis_plus': np.vstack([x['x_dis_plus'] for x in data])
    }

    pct, interior_x_cts, bdy_x_cts, samp_bdy_pts, outliers = separate_bdy_points(loaded_data['x_cts'], args)
    all_unsafe = np.vstack((bdy_x_cts, samp_bdy_pts))

    dataset = {
        'x_cts': interior_x_cts,
        'x_unsafe': all_unsafe,
        'x_dis_minus': loaded_data['x_dis_minus'],
        'x_dis_plus': loaded_data['x_dis_plus']
    }

    plot_chunks(loaded_data['x_cts'][:100000], outliers[:100000], args)

    size = lambda arr: arr.shape[0]

    x = PrettyTable()
    x.field_names = ['Parameter', 'Value']
    x.add_row(['# rollouts', len(data)])
    x.add_row(['Length of each rollout', len(data[0]["x_cts"])])
    x.add_row(['# continuous states', size(loaded_data["x_cts"])])
    x.add_row(['# pre-jump discrete states', size(loaded_data['x_dis_minus'])])
    x.add_row(['# post-jump discrete states', size(loaded_data['x_dis_plus'])])
    x.add_row(['# safe states', size(interior_x_cts)])
    x.add_row(['# unsafe states (rollouts + sampled)', f'{size(bdy_x_cts)} + {size(samp_bdy_pts)} = {size(all_unsafe)}'])
    x.add_row(['Boundary/unsafe state percentage', f'{pct:.3f}'])
    x.align = 'l'
    print(x)

    return dataset

def plot_chunks(x_cts, outliers, args):
    min_vel, max_vel = np.min(x_cts[:, 3]), np.max(x_cts[:, 3])
    bins = list(np.linspace(min_vel, max_vel, num=N_CHUNKS+1, endpoint=True))
    df = pd.DataFrame(x_cts, columns=['Theta_stance', 'Theta_swing', 'Vel_stance', 'Vel_swing'])
    df['Outlier'] = outliers
    df['Chunk'] = df.apply(lambda row: get_chunk(row, bins), axis=1)

    sns.set(style='white', font_scale=1.3)
    fig, axes = plt.subplots(nrows=N_ROWS, ncols=N_COLS, subplot_kw={'projection': '3d'})

    for ch_idx, (i, j) in enumerate(itertools.product(range(N_ROWS), range(N_COLS))):

        ax = axes[i, j]
        curr_df = df[df['Chunk'] == ch_idx + 1]
        in_data, out_data = get_df_data(curr_df)
        
        ax.scatter(in_data[0], in_data[1], in_data[2], color='blue')
        ax.scatter(out_data[0], out_data[1], out_data[2], color='orange')

        if ch_idx == 0:
            ax.set_title(f'$\dot \\theta \leq$ {bins[ch_idx]:.2f}')
            print(f'Chunk: Vel_swing <= {bins[ch_idx]:.2f}', end='')
        elif ch_idx == len(bins) - 1:
            ax.set_title(f'{bins[ch_idx]:.2f} $\leq \dot \\theta$')
            print(f'Chunk: {bins[ch_idx]:.2f} <= Vel_swing', end='')
        else:
            ax.set_title(f'{bins[ch_idx-1]:.2f} $\leq \dot \\theta$$\leq$ {bins[ch_idx]:.2f}')
            print(f'Chunk: {bins[ch_idx-1]:.2f} <= Vel_swing <= {bins[ch_idx]:.2f}', end='')

        print('  ', in_data[0].shape, out_data[0].shape)

    save_path = os.path.join(args.results_dir, 'outlier_chunks.png')
    plt.savefig(save_path)

def get_bdy_points_kdtree(x, thresh, min_num_nbrs):

    tree = KDTree(x)
    dists = tree.query_radius(x, r=thresh, count_only=True)
    return np.array(dists < min_num_nbrs)


def separate_bdy_points(x_cts, args):
    """Extract boundary/interior points using NUTS algorithm."""

    # run NUTS algorithm on continuous states
    outliers = get_bdy_points_kdtree(x_cts, args.nbr_thresh, args.min_num_nbrs)
    pct = 100 * sum([int(x) for x in outliers]) / len(outliers)     
    
    # extract safe/interion and unsafe/boundary states
    interior_x_cts = x_cts[outliers == 0]
    bdy_x_cts = x_cts[outliers == 1]

    # sample additional states near boudary states
    samp_arrays = []
    for _ in range(1):
        samp_bdy_pts = bdy_x_cts + 0.01 * np.random.rand(*bdy_x_cts.shape)
        samp_arrays.append(samp_bdy_pts)
    samp_bdy_pts = np.vstack(samp_arrays)

    return pct, interior_x_cts, bdy_x_cts, samp_bdy_pts, outliers