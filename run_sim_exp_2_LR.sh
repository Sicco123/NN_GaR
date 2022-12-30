#!/usr/bin/env zsh
eval "$(conda shell.bash hook)"
conda activate base

echo Experiment Linear Quantile Regression on SESTAR simulated data is running...
python sim_experiment_2_SESTAR_lin_reg.py
