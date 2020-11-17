import argparse

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
    # (horizon, dt, and fix_left should match those used to collect the training dataset)
    parser.add_argument('--horizon', type=int, default=750,
                            help='Number of steps for each rollout')
    parser.add_argument('--dt', type=float, default=0.01, 
                            help='Time interval between discrete steps')
    parser.add_argument('--fix-left', action='store_true',
                            help='Fix left leg in all initial conditions')
    parser.add_argument('--success-n-steps', type=int, default=5, 
                            help='Number of steps taken to entail a success')

    # other
    parser.add_argument('--report-int', type=int, default=100,
                            help='Print frequency (per epoch) for training')
    parser.add_argument('--reload', action='store_true',
                            help='Reloads neural network from file if argument is provided.')
    parser.add_argument('--reload-path', type=str, 
                            help='Path to saved neural network parameters file (.npy)')














    # # paths to various directories
    # parser.add_argument('--save-path', type=str, 
    #                         help='Path for saving outputs')
    # parser.add_argument('--logdir', default='', type=str,
    #                         help='Directory for tensorboard logs')
    # parser.add_argument('--model-paths', type=str, nargs='*',
    #                         help="Path for model of natural variation")
    
    # # optimization settings and training parameters
    # parser.add_argument('--half-prec', action='store_true', 
    #                         help='Run model in half-precision mode using apex')
    # parser.add_argument('--apex-opt-level', default='O1', type=str, 
    #                         help='opt_level for Apex amp initialization')
    # parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', 
    #                         help='weight decay (default: 1e-4)')
    # parser.add_argument('--init-bn0', action='store_true', 
    #                         help='Intialize running batch norm mean to 0')
    # parser.add_argument('--no-bn-wd', action='store_true', 
    #                         help='Remove batch norm from weight decay')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M', 
    #                         help='Momentum for SGD')
    # parser.add_argument('--data-size', type=int, default=224, 
    #                         help="Size of each image")
    # parser.add_argument('--batch-size', type=int, default=256, 
    #                         help='Training/validation batch size')
    # parser.add_argument('--delta-dim', type=int, default=2, 
    #                         help="dimension of nuisance latent space")

    # # architecture
    # parser.add_argument('--architecture', default='resnet50', type=str, 
    #                         help='Architecture for classifier')
    # parser.add_argument('--pretrained', action='store_true', 
    #                         help='Use pretrained model (only available for torchvision.models)')
    # parser.add_argument('--num-classes', default=1000, type=int, 
    #                         help='Number of classes in datset')
    
    # # dataset
    # parser.add_argument('--dataset', required=True, type=str, choices=['imagenet', 'svhn', 'gtsrb', 'cure-tsr'],
    #                         help='Dataset to use for training/testing classifier.')
    # parser.add_argument('--source-of-nat-var', type=str, 
    #                         help='Source of natural variation')

    # # other parameters
    # parser.add_argument('--print-freq', '-p', default=5, type=int, metavar='N', 
    #                         help='log/print every this many steps (default: 5)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                         help='path to latest checkpoint (default: none)')
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                         help='evaluate model on validation set')
    # parser.add_argument('--short-epoch', action='store_true', 
    #                         help='make epochs short (for debugging)')
    # parser.add_argument('--setup-verbose', action='store_true', 
    #                         help='Print setup messages to console')
    # parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
    #                         help='number of data loading workers (default: 8)')
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                         help='manual epoch number (useful on restarts)')

    args = parser.parse_args()

    return args