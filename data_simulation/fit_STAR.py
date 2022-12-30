import numpy as np
from scipy.optimize import minimize

# Define the NARDL(1,0) model as a function
def star_model(params, x, y):
    delta = params[0]
    gamma = params[1]
    alpha = params[2]
    beta = params[3:7]
    
    y_pred = np.zeros(len(y))
    for t in range(0, len(y)-1):
        exog_coef = delta + gamma/(1+np.exp(alpha+X[t,:]@beta))
        y_pred[t+1] = exog_coef*y[t] 
    
    return y_pred[1:]

# Define the objective function to minimize
def objective(params, x, y):
    y_pred = star_model(params, x, y)
    loss = np.mean((y[1:] - y_pred) ** 2)
    return loss

# Set the initial values for the parameters
init_delta = 0.1
init_gamma = 0.5
init_alpha = 0
init_beta = np.array([0.1,0.2,0.3,0.4])


params_init = np.hstack([init_alpha, init_beta, init_gamma, init_delta])

# Set the optimization constraints
bound_delta = (None,None)
bound_gamma = (None,None)
bound_alpha = (None,None)
bound_beta = [(None, None)]*4
bounds = [bound_delta, bound_gamma, bound_alpha, *bound_beta]

data = np.loadtxt('data_simulation/empirical_data/data_per_country/SWE.csv', delimiter=',', skiprows = 1)
y = data[:,5]
X = data[:,1:5]


# Perform the optimization
result = minimize(fun=objective, x0=params_init, args=(X, y), bounds=bounds)

# Get the optimized parameters
delta_opt = result.x[0]
gamma_opt = result.x[1]
alpha_opt = result.x[2]
beta_opt = result.x[3:7]

# Print the optimized parameters
print('alpha =', alpha_opt)
print('beta =', beta_opt)
print('gamma =', gamma_opt)
print('delta =', delta_opt)
# Plot the predicted and actual values
import matplotlib.pyplot as plt
y_pred = star_model([delta_opt, gamma_opt, alpha_opt, *beta_opt], X, y)
plt.plot(y[1:], label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
