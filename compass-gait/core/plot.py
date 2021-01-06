import pickle
import jax.numpy as jnp
import os

from make_ctrls import get_energy_controller, get_zero_controller, make_safe_controller
from plotting.plotting import plot_phase_port, plot_heatmap
from plotting.movie import make_movie
from collect_data import test_ctrls, find_success_hcbf
from NeuralNet_Dual_Indiv import NeuralNet


def main():

    net = NeuralNet(args.neural_net_dims, args, 'Adam', {'step_size': 0.005})

    loaded_params = jnp.load(args.reload_path, allow_pickle=True)
    learned_h = lambda x: net.forward_indiv(x, loaded_params)

    # make nominal and safe HCBF-QP controllers
    energy_ctrl = get_energy_controller(params=args.cg_params)
    no_ctrl = get_zero_controller()
    safe_ctrl = make_safe_controller(no_ctrl, learned_h, args.cg_params)

    # make plots
    # phase_portrait(learned_h, safe_ctrl, no_ctrl, args)
    heatmap(safe_ctrl, energy_ctrl, learned_h, args, n_trials=5)


def phase_portrait(learned_h, safe_ctrl, zero_ctrl, args):
    """Plot phase portrait and save trajectories to pickle files."""

    safe_traj, baseline_traj = find_success_hcbf(learned_h,  
                                    safe_ctrl, zero_ctrl, args)
    
    make_movie(baseline_traj, 'zero', args)
    make_movie(safe_traj, 'safe', args)

    root = os.path.join(args.results_dir, 'phase-portrait')
    os.makedirs(root, exist_ok=True)

    save_file(safe_traj, os.path.join(root, 'success-hcbf.pkl'))
    save_file(baseline_traj, os.path.join(root, 'failure-energy.pkl'))
    plot_phase_port(safe_traj, args)

def heatmap(safe_ctrl, energy_ctrl, learned_h, args, n_trials=5):
    """Plot heatmap and save trajectories to pickle file."""

    ctrls = {'safe': safe_ctrl, 'energy': energy_ctrl}
    results = test_ctrls(ctrls, n_trials, args)

    root = os.path.join(args.results_dir, 'heatmap')
    os.makedirs(root, exist_ok=True)

    save_file(results, os.path.join(root, 'heatmap_rollouts.pkl'))
    plot_heatmap(learned_h, results)

def save_file(traj, fname):
    """Save object to pickle file."""

    with open(fname, 'wb') as fp:
        pickle.dump(traj, fp)

if __name__ == '__main__':
    main()