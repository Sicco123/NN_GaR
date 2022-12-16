import os
import numpy as np
import pickle

dir = 'simulated_data'
date = '221014'
size = 150

for i in range (1,13):
    tot_size = 0
    for m in range(10000):
        path = f'{dir}/{size}_f_quantiles_{i}/{date}/{m}.csv'
        data = np.loadtxt(path, delimiter =',')
        tot_size += os.path.getsize(path)

        if m == 0:
            appended_data = data
        else:
            appended_data = np.column_stack([appended_data,data])

    path = f'{dir}/{size}_f_quantiles_{i}/{date}.csv'
    #data = np.loadtxt(f'{dir}/{size}_f_quantiles_{i}/{date}.csv', delimiter = ',')
    size_s = os.path.getsize(path)

    with open(f'{dir}/{size}_f_quantiles_{i}/{date}.pickle', 'wb') as handle:
        pickle.dump(appended_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    size_p = os.path.getsize(f'{dir}/{size}_f_quantiles_{i}/{date}.pickle')

    print(size_s, size_p )



    break


