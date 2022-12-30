###########################
# author: Sicco Kooiker
# date: 30-09-2022
# name: data_simulation
# description: This program is used to simulate SESTAR data. This data is then used
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
import matplotlib.pyplot as plt


def simulate_data(train_sample_size, dir_to_store_simulation, burn_in, test_sample_size, day, n, M):
    seed = 42
    np.random.seed(seed)
    # create directory if non existent
    Path(f"{dir_to_store_simulation}/{train_sample_size}/{day}").mkdir(parents=True, exist_ok=True)

    # Set the number of time steps and variables
    n_timesteps = train_sample_size + burn_in + test_sample_size
    n_variables = n

    for m in range(M):
        # Set the initial values for the variables
        x0 = np.random.random(n_variables)

        # Set the coefficients for the nonlinear equations
        a = 1
        b = np.array([0.1,0.2,0.3,0.4]).T
        gamma = 0.1
        phi = 0.8
        # Create an empty array to hold the simulated data
        data = np.zeros((n_timesteps, n_variables))

        # Set the first row of the data array to the initial values
        data[0,:] = x0

        # Simulate the nonlinear equations for each time step
        for t in range(1, n_timesteps):
            data[t,0] = gamma / (1 + np.exp(data[t-1,1:].dot(b) + a)) * data[t-1,0] + np.random.normal(0, 1, 1) 
            data[t,1:] = phi * data[t-1,1:] + np.random.normal(0, 1, n_variables-1)
        # Store the data as pkl file
        with open(f'{dir_to_store_simulation}/{train_sample_size}/{day}/{m}.pkl', 'wb') as f:
            pickle.dump(data, f)
        

def main():
    dir_to_store_simulation = "simulated_SESTAR_data"
    #date = datetime.datetime.now()
    #day = f'{date.strftime("%y")}{date.strftime("%m")}{date.strftime("%d")}'
    date = "221014"

    random.seed(42)
    number_of_samples = 10000
    burn_in = 100
    train_sample_size = 500
    test_sample_size = 200
    quantile_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    n = 5 # number of variables
    M = 10000 # number of simulations

    simulate_data(train_sample_size, dir_to_store_simulation, burn_in, test_sample_size, date, n, M)




if __name__ == '__main__':
    main()