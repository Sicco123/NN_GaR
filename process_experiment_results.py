import numpy as np
from backtests import coverage, tick_loss
import pickle

date = '221014'
experiment_name = 'sim_exp1'
method_1 = 'linear_quantile_regression'
method_2 = 's2s_results'
methods = [method_1, method_2]
M = 500
sample_sizes = [250]
test_size = 200
steps_ahead = [1, 2, 3, 4, 5, 6]
quantile_levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
test_stats = ['coverage', 'tick loss']

results = np.zeros((2,len(quantile_levels),len(steps_ahead), len(test_stats)))
for size in sample_sizes:
    for m in range(M):
        for jdx, step in enumerate(steps_ahead):
            target = np.loadtxt(f'simulated_data/{size}/{date}/{m}.csv', delimiter=",")[-test_size:, 0]
            for zdx, method in enumerate(methods):
                res = np.loadtxt(f'simulation_results/{method}/{size}/{m}/{step}_step_ahead.csv', delimiter=",")[-test_size:,:]

                coverage_stat = coverage.calculate_coverage(res, target)
                coverage_contribution = coverage_stat/M
                results[zdx, :, jdx, 0] += coverage_contribution

                quantile_risk = tick_loss.quantile_risk(res, target, quantile_levels)
                quantile_risk_contribution = quantile_risk/M
                results[zdx, :, jdx, 1] += quantile_risk_contribution

with open(f'simulation_results/{date}_{M}_stats.pkl', "wb") as f:
    pickle.dump(results, f)

print(results[0,:,:,0])
print(results[1,:,:,0])


print(results[0,:,:,1])
print(results[1,:,:,1])

print(results.shape)