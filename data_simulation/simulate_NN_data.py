###########################
# author: Sicco Kooiker
# date: 30-09-2022
# name: data_simulation
# description: This program is used to simulate data. This data is then used
# to compare the linear model with neural network model to estimate quantiles.
###########################

import numpy as np

from scipy.stats import norm
from scipy.stats import t
import datetime
from pathlib import Path
import random
import pickle
import pandas as pd


def simulate_data(train_sample_sizes, dir_to_store_simulation, burn_in, test_sample_size, number_of_predictors,
                  predictor_error_distribution, number_of_samples, day, predictor_model, phi, mqrnn_model,
                    horizon_list, normal_quantiles, quantile_levels, t_quantiles ):

    for train_sample_size in train_sample_sizes:

        Path(f"{dir_to_store_simulation}/{train_sample_size}/{day}").mkdir(parents=True, exist_ok=True)

        # initialization
        simulation_size = burn_in + train_sample_size + test_sample_size
        predictors = np.zeros((simulation_size, number_of_predictors))
        predictor_errors = predictor_error_distribution((simulation_size, number_of_predictors, number_of_samples))



        for b in range(number_of_samples):

            # iterative simulation of predictors
            for t in range(1, simulation_size):
                predictors[t] = predictor_model(predictors[t - 1], phi, predictor_errors[t, :, b])

            # simulated output
            covariate_df = pd.DataFrame(predictors)
            sim_realisations = mqrnn_model.predictions(covariate_df)
            
            realisations_predictors = np.column_stack([sim_realisations[burn_in:,:,3], predictors[burn_in:, :], ])

           

            # store data
            np.savetxt(f'{dir_to_store_simulation}/{train_sample_size}/{day}/{b}.csv', realisations_predictors,
                       delimiter=",")

            print(train_sample_size, b)

def main():
    dir_to_store_simulation = "simulated_NN_data"
    #date = datetime.datetime.now()
    #day = f'{date.strftime("%y")}{date.strftime("%m")}{date.strftime("%d")}'
    date = "221014"

    random.seed(42)
    number_of_samples = 10000
    burn_in = 100
    train_sample_sizes = [150, 250, 500]
    test_sample_size = 200
    quantile_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    
    number_of_predictors = 4
    predictor_model = lambda phi,x_t,e_t : phi*x_t + e_t
    predictor_error_distribution = lambda x : np.random.normal(0,1, x)
    phi = 0.8 # the same for all predictors

    with open(f'{dir_to_store_simulation}/{date}_NN_model.pkl', "rb") as f:
        mqrnn_model = pickle.load(f)
    
    horizon_list = [1,2,3,4,5,6,7]

    normal_quantiles = norm.ppf(quantile_levels)
    t_quantiles = t.ppf(quantile_levels, 10)

    simulate_data(train_sample_sizes, dir_to_store_simulation, burn_in, test_sample_size, number_of_predictors,
                  predictor_error_distribution, number_of_samples, date, predictor_model, phi, mqrnn_model,
                    horizon_list, normal_quantiles, quantile_levels, t_quantiles)




if __name__ == '__main__':
    main()