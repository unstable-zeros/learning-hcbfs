import argparse
from munch import Munch

from utils.parser_utils import add_cg_arguments, munch_cg_params

def get_parser():
    """Parse command line arguments and return args namespace."""

    parser = argparse.ArgumentParser(description='Training a hybrid control barrier function')

    # Dataset paths
    parser.add_argument('--save-data-path', required=False,
                            help='Path to training data pickle file')
    parser.add_argument('--dataset-name', default='trajectories.pkl', type=str,
                            help='Name of dataset to be saved as .pkl file.')
    parser.add_argument('--n-rollouts', type=int, default=100,
                            help='Number of rollouts to use for training')
    parser.add_argument('--horizon', type=int, default=750,
                            help='Number of steps for each rollout')
    parser.add_argument('--dt', type=float, default=0.01, 
                            help='Time interval between discrete steps')
    parser.add_argument('--nominal-ctrl', type=str, default='energy', choices=['energy', 'zero', 'noisy'],
                            help='Nominal controller for collecting data')
    parser.add_argument('--fix-left', action='store_true',
                            help='Fix left leg in all initial conditions')
    parser.add_argument('--success-n-steps', type=int, default=8, 
                            help='Number of steps taken to entail a success')
    parser.add_argument('--n-pert-rollouts', type=int, default=5,
                            help='Number of perturbed rollouts to collect')
    parser.add_argument('--rollout-type', type=str, choices=['iid', 'pert'], default='iid',
                            help='Kind of rollouts to be collected.')
    parser.add_argument('--add-state-noise', action='store_true',
                            help='Adds noise to each state in a rollout.')
    parser.add_argument('--state-noise-level', type=float, default=0.01,
                            help='Noise level for state noise')

    # compass gait parameters
    add_cg_arguments(parser)

    args = parser.parse_args()
    munch_cg_params(args)
    

    return args