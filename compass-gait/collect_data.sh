#!/bin/bash

# Path and filename for saving dataset
export SAVE_DATA_PATH=./
export DATASET_NAME=rollouts.pkl

# Two kinds of rollouts can be collected:
#   * 'iid': all rollouts are generated from iid initial conditions
#   * 'pert': args.n_rollouts iid rollouts are collected.  In addition,
#               args.n_pert_rollouts per iid rollout are collected with
#               perturbed action sequences.
export ROLLOUT_TYPE='iid'

# The nominal (e.g. expert) controller can be one of the following:
#   * 'energy': energy-based controller from Spong et al.
#   * 'zero': controller that always returns zero
#   * 'noisy': controller that returns random noise
export NOMINAL_CTRL='energy'

# Horizon and step size for each rollout
export HORIZON=750
export DELTA_T=0.01

# Specifies number of rollouts and number of perturbed rollouts
export N_ROLLOUTS=3
export N_PERTURB_ROLLOUTS=1

# Number of steps which define a 'success' or 'expert' trajecotry
export SUCCESS_N_STEPS=8

python3 core/collect_data.py --save-data-path $SAVE_DATA_PATH \
    --dataset-name $DATASET_NAME --horizon $HORIZON --dt $DELTA_T \
    --n-rollouts $N_ROLLOUTS --n-pert-rollouts $N_PERTURB_ROLLOUTS \
    --nominal-ctrl $NOMINAL_CTRL --success-n-steps $SUCCESS_N_STEPS \
    --fix-left 