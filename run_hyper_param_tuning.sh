#!/usr/bin/env zsh
echo hyperparameter_tuning is running...
python3 hyperparameter_tuning.py configurations/sestar_150_221014.json 
python3 hyperparameter_tuning.py configurations/sestar_250_221014.json 
python3 hyperparameter_tuning.py configurations/sestar_500_221014.json 
