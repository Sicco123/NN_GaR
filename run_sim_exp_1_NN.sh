#!/usr/bin/env zsh
eval "$(conda shell.bash hook)"
conda activate base

echo Experiment 150 is running...
python3 sim_experiment_1_NN.py configurations/linear_150_221014_no_tune.json

# echo Experiment 250 is running...
# python3 sim_experiment_1_NN.py configurations/linear_250_221014.json

# echo Experiment 500 is running...
# python3 sim_experiment_1_NN.py configurations/linear_500_221014.json