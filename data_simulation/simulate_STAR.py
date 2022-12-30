import numpy as np
import bz2
import pickle
import _pickle as cPickle
from pathlib import Path

dir_to_store_simulation = "simulated_STAR_data"
Path(f"{dir_to_store_simulation}").mkdir(parents=True, exist_ok=True)

T = 750
M = 10000
np.random.seed(42)

for idx in range(M):
  # Generate exogenous variables
  # Define AR model parameters
  phi = 0.8
  sigma = 1
  dim = 4
  # Generate initial value for the AR process
  x0 = np.random.normal(0,sigma, dim)
  x =np.zeros((T, dim))
  x[0] = x0
  # Iterate through each time period
  for t in range(0,T-1):
    # Generate value for the AR process using AR model equation
    x[t+1,:] = phi*x[t,:] + np.random.normal(0,sigma,dim)

  # Define NARDL model parameters
  delta = 0.7
  beta = np.array([-0.3,  0.4,  0.6, 1.1])
  alpha = 0.5
  gamma = -0.7

  # Generate initial values for lagged variables and dependent variable
  y0 = np.random.normal()
  y = np.zeros(T)
  y[0] = y0

  # Iterate through each time period
  for t in range(0, len(y)-1):
        exog_coef = delta + gamma/(1+np.exp(alpha+x[t,:]@beta))
        y[t+1] = exog_coef*y[t] 

  data = np.column_stack([y,x]) 
  with bz2.BZ2File(f'{dir_to_store_simulation}/{idx}' + '.pbz2', 'w') as f: 
    cPickle.dump(data, f)
