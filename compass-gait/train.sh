#!/bin/bash

# datasets
export TRAINING_DATA=./big-dataset.pkl
export N_TRAIN_ROLLOUTS=500

# directory to save all output plots and saved network parameters
export RESULTS_DIR=./results

# architecture and training hyperparameters
export NET_DIMS=(4 32 16 1)
export N_EPOCHS=100

# Lagrange multipliers/coefficients
export LAMBDA_SAFE=5.0
export LAMBDA_UNSAFE=10.0
export LAMBDA_CONT=0.2
export LAMBDA_DISCRETE=0.2
export LAMBDA_GRAD=0.01
export LAMBDA_PARAM=1.0

# Loss margin hyperparameters
export GAMMA_SAFE=0.3
export GAMMA_UNSAFE=0.3
export GAMMA_CONT=0.05
export GAMMA_DISCRETE=0.05

# neighbor sampling NUTS algorithm hyperparameters
export MIN_NUM_NBRS=200
export NBR_THRESH=0.04

python core/train.py --neural-net-dims ${NET_DIMS[@]} --n-epochs $N_EPOCHS \
    --train-data-path $TRAINING_DATA --n-train-rollouts $N_TRAIN_ROLLOUTS \
    --lam-safe $LAMBDA_SAFE --lam-unsafe $LAMBDA_UNSAFE \
    --lam-cnt $LAMBDA_CONT --lam-dis $LAMBDA_DISCRETE \
    --lam-grad $LAMBDA_GRAD --lam-param $LAMBDA_PARAM \
    --gam-safe $GAMMA_SAFE --gam-unsafe $GAMMA_UNSAFE \
    --gam-cnt $GAMMA_CONT --gam-dis $GAMMA_DISCRETE \
    --min-num-nbrs $MIN_NUM_NBRS --nbr-thresh $NBR_THRESH \
    --results-dir $RESULTS_DIR --fix-left 
    # --reload --reload-path ./final_hcbf.npy