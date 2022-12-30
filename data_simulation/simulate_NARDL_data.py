import numpy as np
import bz2
import pickle
import _pickle as cPickle
from pathlib import Path

dir_to_store_simulation = "simulated_NARDL_data"
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
  lag_length = 1
  ar_coefs = [0.2]
  delta = np.array([0.1,-0.4,-0.1,-0.4])
  beta = 10
  alpha = -2
  gamma = np.array([0.1,0.4,0.2,0.1])

  # Generate initial values for lagged variables and dependent variable
  y0 = np.random.normal()
  y = np.zeros(T)
  y[0] = y0

  # Iterate through each time period
  for t in range(0,T-1):
    # Generate value for dependent variable using NARDL model equations
    exog_coef = delta + gamma/(1+np.exp(1+beta*y[t]))
    y[t+1] = ar_coefs[0]*y[t] + x[t+1,:].dot(exog_coef) + np.random.normal()

  data = np.column_stack([y,x]) 
  with bz2.BZ2File(f'{dir_to_store_simulation}/{idx}' + '.pbz2', 'w') as f: 
    cPickle.dump(data, f)
