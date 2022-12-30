import numpy as np
from scipy.optimize import minimize

# Define the NARDL(1,0) model as a function
def nardl_model(params, x, y):
    alpha = params[0]
    beta = params[1]
    gamma = params[2:6]
    delta = params[6:10]
    phi = params[10]
    
    y_pred = np.zeros(len(y))
    for t in range(0, len(y)-1):
        exog_coef = delta + gamma/(1+np.exp(alpha+beta*y[t]))
        y_pred[t+1] = phi*y[t] + x[t+1,:] @ exog_coef 
    
    return y_pred[1:]

# Define the objective function to minimize
def objective(params, x, y):
    y_pred = nardl_model(params, x, y)
    loss = np.mean((y[1:] - y_pred) ** 2)
    return loss

# Set the initial values for the parameters
init_alpha = 0.1
init_beta = 0.5
init_gamma = np.array([0.1,0.3,0.4,0.2])
init_delta = np.array([0.1,0.2,0.3,0.4])
init_phi = 0.5

params_init = np.hstack([init_alpha, init_beta, init_gamma, init_delta, init_phi])

# Set the optimization constraints
bound_alpha = (None, None)
bound_beta = (None, None)
bound_gamma = [(-0.5,0.5)]*4
bound_delta = [(-0.5,0.5)]*4
bound_phi = (-1, 1)
bounds = [bound_alpha, bound_beta, *bound_gamma, *bound_delta, bound_phi]

data = np.loadtxt('data_simulation/empirical_data/data_per_country/SWE.csv', delimiter=',', skiprows = 1)
y = data[:,5]
X = data[:,1:5]


# Perform the optimization
result = minimize(fun=objective, x0=params_init, args=(X, y), bounds=bounds)

# Get the optimized parameters
alpha_opt = result.x[0]
beta_opt = result.x[1]
gamma_opt = result.x[2:6]
delta_opt = result.x[6:10]
phi_opt = result.x[10]

# Print the optimized parameters
print('alpha =', alpha_opt)
print('beta =', beta_opt)
print('gamma =', gamma_opt)
print('delta =', delta_opt)
print('phi =', phi_opt)

# # Plot the predicted and actual values
import matplotlib.pyplot as plt
y_pred = nardl_model([-2,10,0.1,-0.4,-0.1,-0.4,0.1,0.4,0.2,0.1,0.5], X, y)
plt.plot(y[1:], label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
