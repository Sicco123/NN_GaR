import matplotlib.pyplot as plt
import numpy as np
import json
import torch
# import os
# import pickle

# date = '221014'
# num = 448
# sample_size = 250
# y = np.loadtxt(f'simulated_data/{sample_size}/{date}/{num}.csv', delimiter = ',')[:,0]
# one_step_quantiles = np.loadtxt(f'simulated_data/{sample_size}_f_quantiles_1/{date}/{num}.csv', delimiter = ',')[100:,:]
# twelve_step_quantiles = np.loadtxt(f'simulated_data/{sample_size}_f_quantiles_12/{date}/{num}.csv', delimiter = ',')[100:,:]

# one_step_forecasts = np.loadtxt(f'simulation_results/linear_quantile_regression/{sample_size}/{num}/1_step_ahead.csv', delimiter = ',')
# twelve_step_forecasts = np.loadtxt(f'simulation_results/linear_quantile_regression/{sample_size}/{num}/12_step_ahead.csv', delimiter = ',')

# one_step_forecasts_r = np.loadtxt(f'simulation_results/r_results/{sample_size}/{num}/1_step_ahead.csv', delimiter = ',', skiprows= 1)
# twelve_step_forecasts_r = np.loadtxt(f'simulation_results/r_results/{sample_size}/{num}/12_step_ahead.csv', delimiter = ',', skiprows =1)

# params1 = np.loadtxt(f'simulation_results/linear_quantile_regression/{sample_size}/{num}/1_reg_params.csv', delimiter = ',')
# params1r = np.loadtxt(f'simulation_results/r_results/{sample_size}/{num}/1_reg_params.csv', skiprows = 1)

# one_step_forecasts_s2s = np.loadtxt(f'simulation_results/s2s_results/{sample_size}/{num}/1_step_ahead.csv', delimiter = ',')[12:]
# #twelve_step_forecasts_s2s = np.loadtxt(f'simulation_results/s2s_results/{sample_size}/{num}/12_step_ahead.csv', delimiter = ',')[12:]

# x = np.arange(0,len(y)-12,1)

# plt.plot(x,y[12:])
# #plt.plot(x,one_step_quantiles[11:-1,2])
# #plt.plot(x,twelve_step_quantiles[:-12,2])
# plt.plot(x,one_step_forecasts[:,[0,6]])
# #plt.plot(x,twelve_step_forecasts)
# plt.plot(x, one_step_forecasts_s2s[:,[0,6]])
# #plt.plot(x, twelve_step_forecasts_r)
# plt.show()

config = {
    ### output variables
    'horizon_size': 7,
    'horizon_list': [1, 2, 3, 4, 5, 6, 12],
    'quantile_size': 7,
    'tau_vec': [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],

    ### model variables
    'columns': [1],  # name of the endogenous variable
    'hidden_size': 4,
    'context_size': 4,
    'covariate_size': 4 - 1,  # normaly the real data is also used as an extra covariate
    'p1': 20,
    'initialization_prior': 'Gaussian',

    ### training variables
    'lr': 1e-1,
    'layer_size': 1,  # number of lstm layers
    'dropout': 0.0,
    'by_direction': False,
    # True, lstm layers goes in two directions. False, only one direction (data of t+1+h is not used for t+1)
    'batch_size': 1,
    'num_epochs': 150,
    'early_validation_stopping': 8,
    'val_frac': 0.2,
    'load_stored_model': True,  # reload first trained model in training loop
   
    ### storing results
    'store_results': True,
    'store_model': False,
    'dir_to_store_reg_quantiles': 's2s_results',
    'save_nn_name': "s2s_sim_exp1",
    

    ### load data variables
    'M' : 10000,
    'sample_sizes' : [500],
    'test_size' : 200,
    'date' : '221014',
}

with open("configurations/linear_500_221014.json", "w") as jsonfile:
    json.dump(config, jsonfile, indent=2)

with open("configurations/linear_500_221014.json", "r") as jsonfile:
    data = json.load(jsonfile)
    print("Read successful")
print(data)