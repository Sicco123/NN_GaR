import numpy as np
import zipfile


T = 750

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
for t in range(1,T):
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
for t in range(1,T):
  # Generate value for dependent variable using NARDL model equations
  exog_coef = delta + gamma/(1+np.exp(1+beta*y[t]))
  y[t+1] = ar_coefs[0]*y[t] + x[t+1,:].dot(exog_coef) + np.random.normal()
  



# Open a new ZIP file for writing
with zipfile.ZipFile('archive.zip', 'w') as zip:
  # Add a file to the ZIP archive
  zip.write('file.txt')
  
  # Add multiple files to the ZIP archive
  for filename in ['file1.txt', 'file2.txt']:
    zip.write(filename)