import pickle
import jax.numpy as jnp
from jax.numpy.linalg import norm as jnorm
import jax
import os

# from NeuralNet import NeuralNet
# from NeuralNet_Dual import NeuralNet 
from NeuralNet_Dual_Indiv import NeuralNet

from cg_dynamics.compass_gait import Dynamics
from collect_data import get_starting_state, replay_rollout
from make_ctrls import get_energy_controller, get_zero_controller, make_safe_controller
from utils.meters import Saver, AverageMeter
from utils.arg_parser import get_parser
from data.load import kdtree_load_data
from collect_data import test_ctrls, find_success_hcbf
from plotting.plotting import plot_phase_port, plot_heatmap
from plotting.movie import make_movie

def main():

    args = get_parser()
    os.makedirs(args.results_dir, exist_ok=True)

    net = NeuralNet(args.neural_net_dims, args, 'Adam', {'step_size': 0.005})
    
    if args.reload is True:
        # load HCBF from file
        loaded_params = jnp.load(args.reload_path, allow_pickle=True)
        learned_h = lambda x: net.forward_indiv(x, loaded_params)
    else:
        # train HCBF using expert trajectories
        dataset = kdtree_load_data(args)
        learned_h = train_hcbf_primal_dual_indiv(dataset, net, args)

    # make nominal and safe HCBF-QP controllers
    energy_ctrl = get_energy_controller()
    no_ctrl = get_zero_controller()
    safe_ctrl = make_safe_controller(no_ctrl, learned_h)

    # make plots
    # phase_portrait(learned_h, safe_ctrl, no_ctrl, args)
    heatmap(safe_ctrl, energy_ctrl, learned_h, args, n_trials=1000)

def train_hcbf(dataset, net, args):
    """Train a hybrid control barrier function.
    
    Params:
        dataset: Dataset of trajectories.
    """

    params = net.init_params(verbose=True)
    opt_state = net.opt_init(params)

    loss_tracker = AverageMeter('loss', fmt='.3f')
    saver = Saver(args)

    for epoch_idx in range(1, args.n_epochs + 1):
        params = net.get_params(opt_state)

        # compute loss and update tracker
        loss = net.loss(params, dataset)
        consts = net.constraints(params, dataset)

        # update loss tracker
        loss_tracker.update(loss, n=1)

        # do one step of optimization
        opt_state = net.step(epoch_idx, opt_state, dataset)
        
        if epoch_idx % args.report_int == 0:
            print(f'[Epoch: {epoch_idx}/{args.n_epochs}]', end=' ')
            print(f'[Loss: {loss_tracker.val:.3f}]')
            print(f'\t[Safe pct: {consts["safe"]:.3f}]', end=' ')
            print(f'[Unsafe pct: {consts["unsafe"]:.3f}]', end=' ')
            print(f'[Continuous pct: {consts["cnt"]:.3f}]', end=' ')
            print(f'[Discrete pct: {consts["disc"]:.3f}]\n')
            h_safe = net(dataset['x_cts'], params)
            h_unsafe = net(dataset['x_unsafe'], params)
            saver.update(epoch_idx, loss, consts, h_safe, h_unsafe)
            saver.plot()

    # define learned HCBF as python function
    final_params = net.get_params(opt_state)
    learned_h = lambda x: net.forward_indiv(x, final_params)

    # save parameters of trained network to file
    fname = os.path.join(args.results_dir, 'trained_hcbf.npy')
    jnp.save(fname, final_params)

    return learned_h

def train_hcbf_primal_dual(dataset, net, args):
    """Train a hybrid control barrier function.
    
    Params:
        dataset: Dataset of trajectories.
    """

    params = net.init_params(verbose=True)

    opt_state = net.opt_init(params)

    loss_tracker = AverageMeter('loss', fmt='.3f')
    saver = Saver(args)

    for epoch_idx in range(1, args.n_epochs + 1):
        params = net.get_params(opt_state)
        
        # compute loss and update tracker
        loss = net.loss(params, dataset)
        consts = net.constraints(params, dataset)

        # update loss tracker
        loss_tracker.update(loss, n=1)
        
        # do one step of optimization
        opt_state = net.step(epoch_idx, opt_state, dataset)
        net.dual_step(params, dataset)
        
        if epoch_idx % args.report_int == 0:
            print(f'[Epoch: {epoch_idx}/{args.n_epochs}]', end=' ')
            print(f'[Loss: {loss_tracker.val:.3f}]')
            print(f'\t[Safe pct: {consts["safe"]:.3f}]', end=' ')
            print(f'[Unsafe pct: {consts["unsafe"]:.3f}]', end=' ')
            print(f'[Continuous pct: {consts["cnt"]:.3f}]', end=' ')
            print(f'[Discrete pct: {consts["disc"]:.3f}]')
            print(f'\t[λ_safe: {net.dual_vars["λ_safe"]:.3f}]', end=' ')
            print(f'[λ_unsafe: {net.dual_vars["λ_unsafe"]:.3f}]', end=' ')
            print(f'[λ_cnt: {net.dual_vars["λ_cnt"]:.3f}]', end=' ')
            print(f'[λ_dis: {net.dual_vars["λ_dis"]:.3f}]\n')
            h_safe = net(dataset['x_cts'], params)
            h_unsafe = net(dataset['x_unsafe'], params)
            saver.update(epoch_idx, loss, consts, h_safe, h_unsafe, 
                            dual_vars=net.dual_vars)
            saver.plot()

    # define learned HCBF as python function
    final_params = net.get_params(opt_state)
    learned_h = lambda x: net.forward_indiv(x, final_params)

    # save parameters of trained network to file
    fname = os.path.join(args.results_dir, 'trained_hcbf.npy')
    jnp.save(fname, final_params)

    return learned_h

def train_hcbf_primal_dual_indiv(dataset, net, args):
    """Train a hybrid control barrier function.
    
    Params:
        dataset: Dataset of trajectories.
    """

    params = net.init_params(verbose=True)
    net.init_dual_variables(dataset, verbose=True)

    opt_state = net.opt_init(params)

    loss_tracker = AverageMeter('loss', fmt='.3f')
    saver = Saver(args)

    for epoch_idx in range(1, args.n_epochs + 1):
        params = net.get_params(opt_state)
        
        # compute loss and update tracker
        loss = net.loss(params, dataset)
        consts = net.constraints(params, dataset)

        # update loss tracker
        loss_tracker.update(loss, n=1)
        
        # do one step of optimization
        opt_state = net.step(epoch_idx, opt_state, dataset)
        net.dual_step(params, dataset)
        
        if epoch_idx % args.report_int == 0:
            print(f'[Epoch: {epoch_idx}/{args.n_epochs}]', end=' ')
            print(f'[Loss: {loss_tracker.val:.3f}]')
            print(f'\t[Safe pct: {consts["safe"]:.3f}]', end=' ')
            print(f'[Unsafe pct: {consts["unsafe"]:.3f}]', end=' ')
            print(f'[Continuous pct: {consts["cnt"]:.3f}]', end=' ')
            print(f'[Discrete pct: {consts["disc"]:.3f}]')
            print(f'\t[λ_safe: {jnorm(net.dual_vars["λ_safe"]):.3f}]', end=' ')
            print(f'[λ_unsafe: {jnorm(net.dual_vars["λ_unsafe"]):.3f}]', end=' ')
            print(f'[λ_cnt: {jnorm(net.dual_vars["λ_cnt"]):.3f}]', end=' ')
            print(f'[λ_dis: {jnorm(net.dual_vars["λ_dis"]):.3f}]\n')
            h_safe = net(dataset['x_cts'], params)
            h_unsafe = net(dataset['x_unsafe'], params)
            saver.update(epoch_idx, loss, consts, h_safe, h_unsafe, 
                            dual_vars=net.dual_vars)
            saver.plot()

    # define learned HCBF as python function
    final_params = net.get_params(opt_state)
    learned_h = lambda x: net.forward_indiv(x, final_params)

    # save parameters of trained network to file
    fname = os.path.join(args.results_dir, 'trained_hcbf.npy')
    jnp.save(fname, final_params)

    return learned_h

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