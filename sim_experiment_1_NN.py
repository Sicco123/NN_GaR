from pytorch_NCMQRNN.NCMQRNN import NCMQRNN
from pytorch_NCMQRNN.data import NCMQRNN_dataset, read_df
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import os
import pickle
import json

### Magic Numbers
torch.set_num_threads(4)
torch.set_num_interop_threads(4)


def training_loop(config, df):
    ### Prepare Data
    sample_size = config['sample_size']
    m = config['m']
    date = config['date']


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
    net = NCMQRNN(config, config['device'])

    ### Train Model
    net.train(train_dataset, train_val_data)

    ### Get Quantiles
    col_name = config['columns'][0]
    quantile_predictions = net.predictions(covariate_df)#,target_df,  col_name)

    return quantile_predictions, net.validation_losses

def store_quantile_predictions(config, quantile_predictions):
    sample_size = config['sample_size']
    m = config['m']
    if config['store_results']:
        Path(f"simulation_results/{config['dir_to_store_reg_quantiles']}/{sample_size}/{m}").mkdir(parents=True, exist_ok=True)
        for hor in range(config['horizon_size']):
            np.savetxt(f"simulation_results/{config['dir_to_store_reg_quantiles']}/{sample_size}/{m}/{config['horizon_list'][hor]}_step_ahead.csv",
                       quantile_predictions[:, hor, :], delimiter=",")

def delete_all_files_in_folder(dir):

    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def main():
    date = config['date']
    sample_size = config['sample_size']
    test_size = config['test_size']
    for m in range(4954,config['M']):
            config['m'] = m
            config['seed'] += m
            
            ### read data from csv file
            df = pd.read_csv(f"simulated_data/{sample_size}/{date}/{m}.csv", header = None).iloc[:sample_size,:]
            
            if m == 0:
                store_setting = config['load_stored_model']
                config['load_stored_model'] = False

                quantile_predictions, validation_losses = training_loop(config, df)
                
                config['load_stored_model'] = store_setting

            else:
                quantile_predictions, validation_losses = training_loop(config, df)

            store_quantile_predictions(config, quantile_predictions)

            print(m)

    delete_all_files_in_folder('stored_nn_configurations')


if __name__ == '__main__':
    import sys
    _, INPUT = sys.argv
    with open(INPUT ) as f:
        config = json.load(f)

    config["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config["tau_vec"] = np.array(config["tau_vec"])

    main()
