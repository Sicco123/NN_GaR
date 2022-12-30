import numpy as np

def simulate_sinus_path(T, num_of_variables = 4, seed = 42, return_quantiles = False):
    np.random.seed(seed)
    X  = np.random.uniform(-1,1,(num_of_variables,T))
    y = np.zeros(T)

    for i in range(num_of_variables):
        y = y + np.sin(np.pi * (X[i,:] + i/4))/((X[i,:] + i/4)*np.pi)
    
    if not return_quantiles:
        return y, X

### plot 
import matplotlib.pyplot as plt
y, X = simulate_sinus_path(100000, num_of_variables=2)
x1, x2 = X[0], X[1]

### 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c='r', marker='o')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()

