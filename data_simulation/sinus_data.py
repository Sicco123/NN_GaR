import numpy as np

def simulate_sinus_path(T, num_of_variables = 4, seed = 42, return_quantiles = False):
    np.random.seed(seed)
    X  = np.random.uniform(-1,1,(num_of_variables,T))
    y = np.zeros(T)

    for i in range(num_of_variables):
        y = y + np.sin(np.pi * (X[i,:] + i/4))/((X[i,:] + i/4)*np.pi)
    
    if not return_quantiles:
        return y