import os
os.chdir('/Users/siccokooiker/surfdrive/PycharmProjects/NN_GaR')
print(os.getcwd())
from sim_experiment_2_NARDL_S2S import training_loop
from functools import partial
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import Tuner
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from pytorch_NCMQRNN.NCMQRNN import NCMQRNN
import pickle
import sys
import json
_, INPUT = sys.argv
with open(INPUT ) as f:
    config = json.load(f)

num_samples = 500
max_num_epochs = 150
gpus_per_trial = 0
config['seed'] = None


tunable_hyperparams = {
    "p1": tune.choice([5, 10, 20, 40]),
    "context_size": tune.choice([ 1, 2, 4, 8, 12]),
    "lr": tune.choice([1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]),
    "hidden_size": tune.choice([4,8,12])
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config['device'] = device

date = config['date']
sample_size = config['sample_size']
test_size = config['test_size']
m = config['m']
#df = pd.read_csv(f'simulated_data/{sample_size}/{date}/{m}.csv', index_col=None, header=None)
with open(f'simulated_SESTAR_data/{500}/{date}/{m}.pkl', 'rb') as f:
                data = pickle.load(f)
                data = data[-sample_size-test_size:-test_size,:]
                df = pd.DataFrame(data)

config.update(tunable_hyperparams)


def delete_all_files_in_folder(dir):

    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def tuning_training_loop(config):
    os.chdir('/Users/siccokooiker/surfdrive/PycharmProjects/NN_GaR')
    quantile_predictions, validation_losses = training_loop(config, df)

    for val_loss in validation_losses:
        tune.report(loss= val_loss)



scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=100,
    reduction_factor=2)

reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "training_iteration"])

result = tune.run(
    partial(tuning_training_loop),
    resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=True,
    local_dir = os.getcwd())

delete_all_files_in_folder('stored_nn_configurations')


best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))


best_trained_model = NCMQRNN(best_trial.config, device)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)

del best_trial.config['device']

config_str = json.dumps(best_trial.config, indent=2)
with open(INPUT, 'w') as outfile:
    outfile.write(config_str)