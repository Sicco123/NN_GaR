import linear_quantile.model as lq
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time
import multiprocessing
from functools import partial
from prepare_data import prepare_forecast_target, prepare_forecast_predictors

horizon_list = [1,2,3,4,5,6,12]
quantile_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
sample_sizes = [150, 250, 500]
M = 10000
dir_to_store_simulation = "simulated_data"
dir_to_store_reg_quantiles = "linear_quantile_regression"
date = datetime.datetime.now()
day = f'{date.strftime("%y")}{date.strftime("%m")}{date.strftime("%d")}'

test_size = 200

def estimation(h, X_train, X, sample_size, y_star, m):
    X_train_star = prepare_forecast_predictors(horizon_list, h, X_train)
    X_star = prepare_forecast_predictors(horizon_list, h, X)

    model = lq.quantile_regression(quantile_levels=quantile_levels, solver='highs')
    model.fit_multiple(X_train_star, y_star)
    params = model.vparams
    params = np.column_stack([quantile_levels, params])

    reg_quantiles = model.predict_multiple(X_star)

    Path(f"simulation_results/{dir_to_store_reg_quantiles}/{sample_size}/{m}").mkdir(parents=True, exist_ok=True)
    np.savetxt(f"simulation_results/{dir_to_store_reg_quantiles}/{sample_size}/{m}/{h}_step_ahead.csv", reg_quantiles,
               delimiter=",")
    np.savetxt(f"simulation_results/{dir_to_store_reg_quantiles}/{sample_size}/{m}/{h}_reg_params.csv", params,
               delimiter=",")



def main():
    start = time.time()
    for m in range(0,M):

        for sample_size in sample_sizes:
            data = np.loadtxt(f'simulated_data/{sample_size}/{m}_221014.csv', delimiter=',')

            y = data[:, 0]
            X = data[:, 1:]
            y_train = data[:-test_size,0]
            X_train = data[:-test_size,1:]

            X = np.column_stack([np.ones(X.shape[0]), X])
            X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])

            y_star = prepare_forecast_target(horizon_list, y_train)

            pool = multiprocessing.Pool(processes= 4)
            multiprocessing_func = partial(estimation, X_train = X_train, X= X, sample_size = sample_size, y_star = y_star, m=m)
            pool.map(multiprocessing_func, horizon_list)
            pool.close()

            time_it = time.time()
            seconds = time_it - start
            minutes = (seconds - seconds % 60) / 60
            print(f'm: {m}    Sample size: {sample_size}   Time in minutes: {int(minutes)}')




if __name__ == '__main__':
    main()


