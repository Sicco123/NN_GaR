#%%
import numpy as np

T =750 
X  = np.random.uniform(-1,1,(4,T))
y = np.zeros(T)

y = np.sin(np.pi * (X[0,:]))/((X[0,:])*np.pi) 
y = y #+ (np.exp(0.5-1/2*X[0,:]-1/2*X[1,:]))*np.random.normal(0,1,T)

# plot the data
import matplotlib.pyplot as plt
time = np.arange(0,T)
plt.scatter( X[0,:], y)
plt.show()


# %%
