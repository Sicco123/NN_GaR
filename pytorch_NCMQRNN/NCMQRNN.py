import torch
from .Encoder import Encoder
from .Decoder import GlobalDecoder, Penalizer
from .train_func import train_fn
from .data import NCMQRNN_dataset
from pytorch_NCMQRNN.l1_penalization_layer import non_cross_transformation
from pathlib import Path
from os import path

class NCMQRNN(object):
    """
    This class holds the encoder and the decoder.
    """
    def __init__(self, 
                config, target,
                device):

        self.device = device
        self.horizon_size = config['horizon_size']
        self.horizon_list = config['horizon_list']
        self.hidden_size = config['hidden_size']
        self.layer_size = config['layer_size']
        self.dropout = config['dropout']
        self.quantile_size = config['quantile_size']
        self.quantiles = config['tau_vec']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.early_validation_stopping = config['early_validation_stopping'] if 'early_validation_stopping' in config.keys() else self.num_epochs
        self.covariate_size = config['covariate_size']
        self.context_size = config['context_size']
        self.p1 = config['p1']
        self.initialization_prior = config['initialization_prior']
        self.by_direction = config['by_direction']
        self.name = config['save_nn_name']
        self.m = config['m']
        load_stored = config['load_stored_model']

        Path(f"stored_nn_configurations").mkdir(parents=True, exist_ok=True)

        self.encoder = Encoder(horizon_size=self.horizon_size,
                               covariate_size=self.covariate_size,
                               hidden_size=self.hidden_size,
                               dropout=self.dropout,
                               layer_size=self.layer_size,
                               by_direction=self.by_direction,
                               device=device)
        
        self.gdecoder = GlobalDecoder(hidden_size=self.hidden_size,
                                    covariate_size=self.covariate_size,
                                    horizon_size=self.horizon_size,
                                    horizon_list = self.horizon_list,
                                    context_size=self.context_size)

        self.penalizer = Penalizer(quantile_size=self.quantile_size,
                                    context_size=self.context_size,
                                    quantiles=self.quantiles,
                                    horizon_size=self.horizon_size,
                                    horizon_list = self.horizon_list,
                                    initialization_prior = self.initialization_prior,
                                    target = target)
        self.encoder.double()
        self.gdecoder.double()
        self.penalizer.double()

        if load_stored:
            self.load((f'{self.name} + 0')) # This loads the weights of the last optimized model. This weight initialization might significantly increase computation time.
    
    def train(self, dataset:NCMQRNN_dataset, val_data:NCMQRNN_dataset):
        
        train_fn(encoder=self.encoder, 
                gdecoder=self.gdecoder,
                penalizer = self.penalizer,
                train_data=dataset,
                val_data = val_data,
                lr=self.lr,
                batch_size=self.batch_size,
                num_epochs=self.num_epochs,
                early_stop_crit = self.early_validation_stopping,
                name=self.name + self.m,
                p1 = self.p1,
                horizon_size = self.horizon_size,
                device=self.device)
        print("training finished")

    def load(self, name):

        boolean_encoder_path = path.exists(f'stored_nn_configurations/{name}_saved_encoder.pth')
        boolean_gdecoder_path = path.exists(f'stored_nn_configurations/{name}_saved_gdecoder.pth')
        boolean_penalizer_path = path.exists(f'stored_nn_configurations/{name}_saved_penalizer.pth')

        if boolean_encoder_path and boolean_gdecoder_path and boolean_penalizer_path and self.m > 0:
            self.encoder.load_state_dict(torch.load(f'stored_nn_configurations/{name}_saved_encoder.pth'))
            self.encoder.eval()

            self.gdecoder.load_state_dict(torch.load(f'stored_nn_configurations/{name}_saved_gdecoder.pth'))
            self.gdecoder.eval()

            self.penalizer.load_state_dict(torch.load(f'stored_nn_configurations/{name}_saved_penalizer.pth'))
            self.encoder.eval()

    def predict(self,train_target_df, train_covariate_df, col_name):

        input_target_tensor = torch.tensor(train_target_df[[col_name]].to_numpy())
        full_covariate = train_covariate_df.to_numpy()
        full_covariate_tensor = torch.tensor(full_covariate)


        input_target_tensor = input_target_tensor.to(self.device)
        full_covariate_tensor = full_covariate_tensor.to(self.device)

        with torch.no_grad():
            input_target_covariate_tensor = torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor, dim= 0) #[1, seq_len, 1+covariate_size]
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1,0,2) #[seq_len, 1, 1+covariate_size]
            print(f"input_target_covariate_tensor shape: {input_target_covariate_tensor.shape}")
            outputs = self.encoder(input_target_covariate_tensor) #[seq_len,1,hidden_size]
            hidden = torch.unsqueeze(outputs[-1],dim=0) #[1,1,hidden_size]


            print(f"hidden shape: {hidden.shape}")
            gdecoder_input = hidden #[1,1, hidden + covariate_size* horizon_size]
            gdecoder_output = self.gdecoder(gdecoder_input) #[1,1,(horizon_size+1)*context_size]
            penalizer_output, loss = self.penalizer(gdecoder_output)



            penalizer_output = penalizer_output.view(self.horizon_size,self.quantile_size)

            i = 1
            for parameter in self.penalizer.parameters():
                if i % 2 != 0:
                    delta_coef_matrix = parameter
                    i+=1
                elif i % 2 == 0 :
                    delta_0_matrix = parameter
                    penalizer_output[int(i/2-1),:] = non_cross_transformation(penalizer_output[int(i/2)-1,:], delta_coef_matrix, delta_0_matrix)

            output_array = penalizer_output.cpu().numpy()


            return output_array

    def predictions(self,target_df, covariate_df, col_name):

        input_target_tensor = torch.tensor(target_df[[col_name]].to_numpy())
        full_covariate = covariate_df.to_numpy()
        full_covariate_tensor = torch.tensor(full_covariate)


        input_target_tensor = input_target_tensor.to(self.device)
        full_covariate_tensor = full_covariate_tensor.to(self.device)

        with torch.no_grad():
            input_target_covariate_tensor = full_covariate_tensor#torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
            input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor, dim= 0) #[1, seq_len, 1+covariate_size]
            input_target_covariate_tensor = input_target_covariate_tensor.permute(1,0,2) #[seq_len, 1, 1+covariate_size]

            outputs = self.encoder(input_target_covariate_tensor) #[seq_len,1,hidden_size]  # input_target_covariate_tensor use this to use the real data as a covariate

            gdecoder_input = outputs #[1,1, hidden + covariate_size* horizon_size]
            gdecoder_output = self.gdecoder(gdecoder_input) #[1,1,(horizon_size+1)*context_size]


            penalizer_output, loss = self.penalizer(gdecoder_output)
            penalizer_output = penalizer_output.view(len(target_df), self.horizon_size,self.quantile_size)


            i = 1
            for parameter in self.penalizer.parameters():
                if i % 2 != 0:
                    delta_coef_matrix = parameter
                    i+=1
                elif i % 2 == 0 :
                    delta_0_matrix = parameter
                    penalizer_output[int(i/2-1),:] = non_cross_transformation(penalizer_output[int(i/2)-1,:], delta_coef_matrix, delta_0_matrix)

            output_array = penalizer_output.detach().cpu().numpy()


            return output_array