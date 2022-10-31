import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

date = '221014'
num = 448
sample_size = 250
y = np.loadtxt(f'simulated_data/{sample_size}/{date}/{num}.csv', delimiter = ',')[:,0]
one_step_quantiles = np.loadtxt(f'simulated_data/{sample_size}_f_quantiles_1/{date}/{num}.csv', delimiter = ',')[100:,:]
twelve_step_quantiles = np.loadtxt(f'simulated_data/{sample_size}_f_quantiles_12/{date}/{num}.csv', delimiter = ',')[100:,:]

one_step_forecasts = np.loadtxt(f'simulation_results/linear_quantile_regression/{sample_size}/{num}/1_step_ahead.csv', delimiter = ',')
twelve_step_forecasts = np.loadtxt(f'simulation_results/linear_quantile_regression/{sample_size}/{num}/12_step_ahead.csv', delimiter = ',')

one_step_forecasts_r = np.loadtxt(f'simulation_results/r_results/{sample_size}/{num}/1_step_ahead.csv', delimiter = ',', skiprows= 1)
twelve_step_forecasts_r = np.loadtxt(f'simulation_results/r_results/{sample_size}/{num}/12_step_ahead.csv', delimiter = ',', skiprows =1)

params1 = np.loadtxt(f'simulation_results/linear_quantile_regression/{sample_size}/{num}/1_reg_params.csv', delimiter = ',')
params1r = np.loadtxt(f'simulation_results/r_results/{sample_size}/{num}/1_reg_params.csv', skiprows = 1)

one_step_forecasts_s2s = np.loadtxt(f'simulation_results/s2s_results/{sample_size}/{num}/1_step_ahead.csv', delimiter = ',')[12:]
#twelve_step_forecasts_s2s = np.loadtxt(f'simulation_results/s2s_results/{sample_size}/{num}/12_step_ahead.csv', delimiter = ',')[12:]

x = np.arange(0,len(y)-12,1)

plt.plot(x,y[12:])
#plt.plot(x,one_step_quantiles[11:-1,2])
#plt.plot(x,twelve_step_quantiles[:-12,2])
plt.plot(x,one_step_forecasts[:,[0,6]])
#plt.plot(x,twelve_step_forecasts)
plt.plot(x, one_step_forecasts_s2s[:,[0,6]])
#plt.plot(x, twelve_step_forecasts_r)
plt.show()

