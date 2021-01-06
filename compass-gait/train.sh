#!/bin/bash

# datasets
export TRAINING_DATA=./vary-hip-mass/all-trajs-shuffled.pkl
# export TRAINING_DATA=./big-dataset.pkl
export N_TRAIN_ROLLOUTS=500

# directory to save all output plots and saved network parameters
export RESULTS_DIR=./robust-results-hip-0.1

# Compass gait parameters
export HIP_MASS=10.0                # kg
export LEG_MASS=5.0                 # kg
export LEG_LENGTH=1.0               # m
export LEG_CENTER_OF_MASS=0.5       # m
export GRAVITY=9.81                 # m/s^2
export SLOPE=0.0525                 # radians

# architecture and training hyperparameters
export NET_DIMS=(4 32 16 1)
export N_EPOCHS=30000

# training algorithm
#   * 'primal': fix dual variables as hyperparameters and train by 
#           minimizing the empirical Lagrangian
#   * 'avg-primal-dual': use one dual variable for each type constraint
#           and update dual variables via dual ascent
#   * 'indiv-primal-dual': use one dual variable for each individual
#           constraint and update all dual variables via dual ascent
export train_mode='indiv-primal-dual'

# Lagrange multipliers/coefficients (only used in 'primal' mode)
export LAMBDA_SAFE=5.0
export LAMBDA_UNSAFE=10.0
export LAMBDA_CONT=0.2
export LAMBDA_DISCRETE=0.2
export LAMBDA_GRAD=0.01
export LAMBDA_PARAM=1.0

# Loss margin hyperparameters (used in all three training modes)
export GAMMA_SAFE=0.3
export GAMMA_UNSAFE=0.3
export GAMMA_CONT=0.05
export GAMMA_DISCRETE=0.05

# neighbor sampling NUTS algorithm hyperparameters
export MIN_NUM_NBRS=200
# export NBR_THRESH=0.04        # original HCBF
export NBR_THRESH=0.07          # hip mass HCBF
# export NBR_THRESH=0.05

export CUDA_VISIBLE_DEVICES=3

python3 core/train.py --neural-net-dims ${NET_DIMS[@]} --n-epochs $N_EPOCHS \
    --train-data-path $TRAINING_DATA --n-train-rollouts $N_TRAIN_ROLLOUTS \
    --lam-safe $LAMBDA_SAFE --lam-unsafe $LAMBDA_UNSAFE \
    --lam-cnt $LAMBDA_CONT --lam-dis $LAMBDA_DISCRETE \
    --lam-grad $LAMBDA_GRAD --lam-param $LAMBDA_PARAM \
    --gam-safe $GAMMA_SAFE --gam-unsafe $GAMMA_UNSAFE \
    --gam-cnt $GAMMA_CONT --gam-dis $GAMMA_DISCRETE \
    --min-num-nbrs $MIN_NUM_NBRS --nbr-thresh $NBR_THRESH \
    --results-dir $RESULTS_DIR --fix-left --hip-mass $HIP_MASS \
    --leg-mass $LEG_MASS --leg-length $LEG_LENGTH \
    --leg-cent-of-mass $LEG_CENTER_OF_MASS \
    --gravity $GRAVITY --slope $SLOPE \
    --reload --reload-path robust-results-hip-0.1/trained_hcbf.npy
