from pytorch_NCMQRNN.NCMQRNN import NCMQRNN
from pytorch_NCMQRNN.data import NCMQRNN_dataset, read_df
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import os


### Magic Numbers
config = {
'horizon_size' : 7,
'hidden_size' : 12,
'horizon_list' : [1,2,3,4,5,6,12],
'tau_vec' : np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]),
'quantile_size' : 7,
'columns' : [1],
'dropout' : 0.0,
'layer_size' : 1,
'by_direction' : False,
'lr' : 5e-2,
'batch_size' : 1,
'num_epochs' : 150,
'early_validation_stopping' : 15,
'context_size' : 10,
'covariate_size': 4 -1, #normaly the real data is also used as an extra covariate
'p1' : 10,
'val_frac':0.2,
'initialization_prior':'Gaussian',
'dir_to_store_reg_quantiles' : 's2s_results',
'save_nn_name' : "s2s_sim_exp1",
'load_stored_model': True
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

test_size = 200

M = 500
sample_sizes = [250] #[150, 250, 500]
date = '221014'


def training_loop(config):
    ### Prepare Data
    sample_size = config['sample_size']
    m = config['m']
    df = pd.read_csv(f'simulated_data/{sample_size}/{date}/{m}.csv', index_col=None, header=None)

    train_size = sample_size - int(config['val_frac'] * sample_size)
    val_size = int(config['val_frac'] * sample_size)

    prepared_data = read_df(df, 0, train_size, val_size, normalize=True)
    target_df = prepared_data[0]
    covariate_df = prepared_data[1]
    train_target_df = prepared_data[2]
    train_covariate_df = prepared_data[3]
    train_val_target_df = prepared_data[4]
    train_val_covariate_df = prepared_data[5]

    train_dataset = NCMQRNN_dataset(train_target_df, train_covariate_df, config['horizon_size'])
    train_val_data = NCMQRNN_dataset(train_val_target_df, train_val_covariate_df, config['horizon_size'])

    ### Set Model
    net = NCMQRNN(config, train_target_df, device)

    ### Train Model
    net.train(train_dataset, train_val_data)

    ### Get Quantiles
    col_name = 1
    quantile_predictions = net.predictions(target_df, covariate_df, col_name)

    Path(f"simulation_results/{config['dir_to_store_reg_quantiles']}/{sample_size}/{m}").mkdir(parents=True, exist_ok=True)
    for hor in range(config['horizon_size']):
        np.savetxt(f"simulation_results/{config['dir_to_store_reg_quantiles']}/{sample_size}/{m}/{hor}_step_ahead.csv",
                   quantile_predictions[:, hor - 1, :], delimiter=",")

def delete_all_files_in_folder(dir):

    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def main():

    for sample_size in sample_sizes:
        config['sample_size'] = sample_size


        for m in range(0,M):
                config['m'] = m

                if m == 0:
                    store_setting = config['load_stored_model']
                    config['load_stored_model'] = False
                    training_loop(config)

                    config['load_stored_model'] = store_setting

                else:
                    training_loop(config)


                print(m)

        delete_all_files_in_folder('stored_nn_configurations')


if __name__ == '__main__':
    main()