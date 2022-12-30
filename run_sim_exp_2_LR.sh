#!/usr/bin/env zsh
eval "$(conda shell.bash hook)"
conda activate base

echo Experiment Linear Quantile Regression on STAR simulated data is running...
python sim_experiment_2_STAR_lin_reg.py
