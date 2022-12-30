#!/usr/bin/env zsh
eval "$(conda shell.bash hook)"
conda activate base

echo Experiment 150 is running...
python3 sim_experiment_2_STAR.py configurations/star_150_221014.json

echo Experiment 250 is running...
python3 sim_experiment_2_STAR.py configurations/star_250_221014.json

echo Experiment 500 is running...
python3 sim_experiment_2_STAR.py configurations/star_500_221014.json