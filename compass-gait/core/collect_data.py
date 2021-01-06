import numpy as np
import pickle
import os

from utils.data_arg_parser import get_parser
from make_ctrls import *
from cg_dynamics.rollout_utils import replay_rollout
from cg_dynamics.environment import CompassGaitEnv
from cg_dynamics.dynamics import CG_Dynamics
from collectors import get_starting_state, rollout_collector

def main():
    args = get_parser()

    # np.random.seed(0)

    if args.nominal_ctrl == 'energy':
        ctrl = get_energy_controller(args.cg_params)
    elif args.nominal_ctrl == 'zero':
        ctrl = get_zero_controller()
    elif args.nominal_ctrl == 'noisy':
        ctrl = get_noisy_contorller()

    # create agent and compass gait environment
    agent = CG_Dynamics(args.cg_params)
    cg_envir = CompassGaitEnv(dt=args.dt, horizon=args.horizon, agent=agent)

    # collect rollouts
    if args.rollout_type == 'iid':
        trajs = iid_rollouts(ctrl, cg_envir, args, expert=True)

    elif args.rollout_type == 'pert':
        trajs = perturb_ctrl_rollouts(ctrl, cg_envir, args, expert=True)

    # save rollouts to file
    os.makedirs(args.save_data_path, exist_ok=True)
    data_path = os.path.join(args.save_data_path, args.dataset_name)
    with open(data_path, "wb") as fp:
        pickle.dump(trajs, fp)


def iid_rollouts(ctrl, cg_envir, args, n_rollouts=None, expert=True, h=None):
    """Collect IID rollouts from different initial conditions."""

    if n_rollouts is None: n_rollouts = args.n_rollouts

    # get_rollout is a function that will collect a rollout
    get_rollout = rollout_collector(ctrl, cg_envir, add_noise=args.add_state_noise,
                        noise_level=args.state_noise_level)
    
    trajectories = []
    while len(trajectories) < n_rollouts:
        ic = get_starting_state(fix_left=args.fix_left)
        num_steps, action_seq, noise_seq = get_rollout(ic)
        traj = replay_rollout(ic, cg_envir, action_seq, noise_seq, h=h)
        
        if should_add_traj(expert, num_steps, args) is True:
            trajectories.append(traj)

    return trajectories

def perturb_ctrl_rollouts(ctrl, cg_envir, args, expert=True):
    """Collect rollouts by perturbing action sequences."""

    def noisy_seq(seq):
        """Add uniformly generated noise to action sequence."""

        seq = np.array(seq)
        return list(seq + np.random.uniform(size=seq.shape, low=-0.2, high=0.2))

    def replay_with_noise(init_traj, ic):
        """Replay trajectory with noisy action sequence."""

        orig_action_seq = init_traj['action_seq']
        orig_noise_seq = init_traj['noise_seq']
        pert_action_seq = noisy_seq(orig_action_seq)
        return replay_rollout(ic, cg_envir, pert_action_seq, orig_noise_seq)
    
    trajectories = []
    while len(trajectories) < args.n_rollouts * args.n_pert_rollouts:

        # collect an initial rollout
        print('Searching for initial trajectory...')
        init_traj = iid_rollouts(ctrl, cg_envir, args, n_rollouts=1, expert=expert)[0]
        ic = init_traj['init_state']
        trajectories.append(init_traj)
        print('...found initial trajectory.\n')

        # collect trajectories with perturbed action seqs
        pert_idx = 0
        while pert_idx < args.n_pert_rollouts:
            traj = replay_with_noise(init_traj, ic)
            num_steps = traj['n_steps']

            if should_add_traj(expert, num_steps, args) is True:
                print(f'Found perturbed trajectory: {pert_idx}')
                trajectories.append(traj)
                pert_idx += 1
        print('')

    return trajectories
        
def should_add_traj(expert, num_steps, args):
    """Logic that determines whether trajectory should be added to list."""

    if expert is True and num_steps >= args.success_n_steps:
        return True
    elif expert is False:
        return True
    return False





if __name__ == '__main__':
    main()