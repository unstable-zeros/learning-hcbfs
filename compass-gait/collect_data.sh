#!/bin/bash

# Path and filename for saving dataset
export SAVE_DATA_PATH=./additive-noise
export DATASET_NAME=level-0.25-pert-device-3.pkl

# Compass gait parameters
export HIP_MASS=10.0                # kg
export LEG_MASS=5.0                 # kg
export LEG_LENGTH=1.0               # m
export LEG_CENTER_OF_MASS=0.5       # m
export GRAVITY=9.81                 # m/s^2
export SLOPE=0.0525                 # radians

# Two kinds of rollouts can be collected:
#   * 'iid': all rollouts are generated from iid initial conditions
#   * 'pert': args.n_rollouts iid rollouts are collected.  In addition,
#               args.n_pert_rollouts per iid rollout are collected with
#               perturbed action sequences.
export ROLLOUT_TYPE='pert'

# Specifies number of rollouts and number of perturbed rollouts
export N_ROLLOUTS=75
export N_PERTURB_ROLLOUTS=2

# Number of steps which define a 'success' or 'expert' trajectory
export SUCCESS_N_STEPS=8

# The nominal (e.g. expert) controller can be one of the following:
#   * 'energy': energy-based controller from Spong et al.
#   * 'zero': controller that always returns zero
#   * 'noisy': controller that returns random noise
export NOMINAL_CTRL='energy'

# Horizon and step size for each rollout
export HORIZON=750
export DELTA_T=0.01

# State noise level
export NOISE_LEVEL=0.25

export CUDA_VISIBLE_DEVICES=3

python3 core/collect_data.py --save-data-path $SAVE_DATA_PATH \
    --dataset-name $DATASET_NAME --horizon $HORIZON --dt $DELTA_T \
    --n-rollouts $N_ROLLOUTS --n-pert-rollouts $N_PERTURB_ROLLOUTS \
    --nominal-ctrl $NOMINAL_CTRL --success-n-steps $SUCCESS_N_STEPS \
    --fix-left --rollout-type $ROLLOUT_TYPE \
    --hip-mass $HIP_MASS --leg-mass $LEG_MASS \
    --leg-length $LEG_LENGTH --leg-cent-of-mass $LEG_CENTER_OF_MASS \
    --gravity $GRAVITY --slope $SLOPE