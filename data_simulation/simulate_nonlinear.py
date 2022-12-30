import numpy as np

def simulate_nonlinear(N, alpha = 0.75, seed = 42, return_quantiles = False):
    np.random.seed(seed)
    delta = 0.7
    gamma = -0.7
    alpha = 0.5
    beta = np.array([0.5, 3.5])
    x1  = np.random.uniform(0,1,(N))
    x2 = np.random.uniform(0,1,(N))
    y = np.zeros(N)
    errors = np.exp( 1-x1-x2)/10*np.random.normal(0,1,N)
    y = delta + gamma/(1 + np.exp(alpha + beta[0]*x1 + beta[1]*x2))#+errors
    
    if not return_quantiles:
        return y, x1, x2

### plot 
import matplotlib.pyplot as plt
y, x1, x2 = simulate_nonlinear(5000 )

# inverse normal cdf
border95 = 1.96*np.exp(1-x1-x2)/20 + y
border05 = -1.96*np.exp(1-x1-x2)/20 + y

### 3D plot
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x1, x2, y, c='r', marker='o')
# ax.set_xlabel('L')
# ax.set_ylabel('K')
# ax.set_zlabel('Y')
# plt.show()

### 3D surface plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(x1, x2, y, cmap=plt.cm.viridis, linewidth=0.2, alpha=0.9, label='y')
ax.plot_trisurf(x1, x2, border95, cmap= plt.cm.viridis, linewidth=0.2, alpha=0.6, label='95%% confidence interval')
ax.plot_trisurf(x1, x2, border05, cmap=plt.cm.viridis, linewidth=0.2, alpha=0.6, label='5%% confidence interval')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Y')
### make legend in 3D plot

plt.show()


