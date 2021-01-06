import argparse

from utils.parser_utils import add_cg_arguments, munch_cg_params

def get_parser():
    """Parse command line arguments and return args namespace."""

    parser = argparse.ArgumentParser(description='Training a hybrid control barrier function')

    # Dataset paths
    parser.add_argument('--train-data-path', required=False,
                            help='Path to training data pickle file')
    parser.add_argument('--test-data-path', required=False,
                            help='Path to test data pickle file')
    parser.add_argument('--n-train-rollouts', type=int, default=100,
                            help='Number of rollouts to use for training')
    parser.add_argument('--results-dir', type=str, default='./results',
                            help='Path to save all outputs')

    # optimization settings and training parameters
    parser.add_argument('--neural-net-dims', type=int, nargs='*',
                            help='Dimensions of neural network to train')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], 
                            help='Optimization algorithm to use')
    parser.add_argument('--learning-rate', type=float, default=0.005, 
                            help='Learning rate for optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, 
                            help='Momentum for SGD')
    parser.add_argument('--n-epochs', type=int, default=20000,
                            help='Number of epochs for training the HCBF')

    # hybrid CBF hyperparameters
    parser.add_argument('--lam-safe', type=float, default=5.0,
                            help='Lagrange multiplier for safe states loss')
    parser.add_argument('--lam-unsafe', type=float, default=5.0,
                            help='Lagrange multiplier for unsafe states loss')
    parser.add_argument('--lam-cnt', type=float, default=0.5,
                            help='Lagrange multiplier for continuous states loss')
    parser.add_argument('--lam-dis', type=float, default=0.5,
                            help='Lagrange multiplier for discrete states loss')
    parser.add_argument('--lam-grad', type=float, default=0.01,
                            help='Lagrange multiplier for penalty on gradient of h(x)')
    parser.add_argument('--lam-param', type=float, default=0.01,
                            help='Lagrange multiplier for penalty on the size of the weights of h(x)')
    parser.add_argument('--gam-safe', type=float, default=0.3,
                            help='Margin value for safe loss')
    parser.add_argument('--gam-unsafe', type=float, default=0.3,
                            help='Margin value for safe loss')
    parser.add_argument('--gam-cnt', type=float, default=0.05,
                            help='Margin value for continuous loss')
    parser.add_argument('--gam-dis', type=float, default=0.05,
                            help='Margin value for discrete loss')

    # boundary sampling hyperparameters
    parser.add_argument('--min-num-nbrs', type=int, default=200,
                            help='Minimum numbers of neighbors for neighbor sampling algorithm')
    parser.add_argument('--nbr-thresh', type=float, default=0.08,
                            help='Neighbor threshold for neighbor sampling algorithm')

    # compass gait parameters 
    parser.add_argument('--horizon', type=int, default=750,
                            help='Number of steps for each rollout')
    parser.add_argument('--dt', type=float, default=0.01, 
                            help='Time interval between discrete steps')
    parser.add_argument('--fix-left', action='store_true',
                            help='Fix left leg in all initial conditions')
    parser.add_argument('--success-n-steps', type=int, default=5, 
                            help='Number of steps taken to entail a success')

    # other
    parser.add_argument('--report-int', type=int, default=10,
                            help='Print frequency (per epoch) for training')
    parser.add_argument('--reload', action='store_true',
                            help='Reloads neural network from file if argument is provided.')
    parser.add_argument('--reload-path', type=str, 
                            help='Path to saved neural network parameters file (.npy)')

    add_cg_arguments(parser)

    args = parser.parse_args()
    munch_cg_params(args)

    return args