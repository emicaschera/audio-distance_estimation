import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa import STFT
from pytorch_lightning import LightningModule
import pandas as pd
from asteroid_filterbanks import STFTFB, make_enc_dec, transforms
    
class SeldNet(nn.Module):
    def __init__(self, kernels, n_grus, features_set, att_conf, n_filters, kernel_size, encoder_type, sampling_rate):
        super(SeldNet, self).__init__()
        self.sampling_rate = sampling_rate
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.encoder_type = encoder_type
        self.n_fft = 512
        self.hop_length = 256
        self.nb_cnn2d_filt = 128
        self.pool_size = [8, 8, 2]
        self.rnn_size = [128, 128]
        self.fnn_size = 128
        self.kernels = kernels
        self.n_grus = n_grus
        self.features_set = features_set
        self.att_conf = att_conf

        padding = 256
        if self.encoder_type == "param_sinc":
            padding = 257

        # create encoder and decoder
        #self.encoder, _ = make_enc_dec(self.encoder_type, self.n_filters, self.kernel_size, sample_rate=16000, padding=padding)

        #custom encoder
        self.encoder = nn.Sequential(
             # Primo blocco: Conv1D -> BatchNorm1D -> ELU

            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),  # Kernel size ridotto a 5
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            #nn.Dropout(0.1),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # Adattato per mantenere le dimensioni
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            #nn.Dropout(0.1),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            #nn.Dropout(0.1),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            #nn.Dropout(0.1)
                            )

        # kernels "freq" [1, 3]
        self.kernels = (1,3)
        
        # self.STFT = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

        # feature set
       # self.data_in = [3, (10*16000+self.n_filters)//(self.stride) - 1,  int(self.n_filters/2)]

        # da rendere parametrica
        self.data_in = [3, (10*16000+self.n_fft)//(self.n_fft - self.hop_length) - 1,  int(self.n_fft/2)]

        
        # ATTENTION MAP False, "onSpec", "onAll"
        self.heatmap = nn.Sequential(
                nn.Conv2d(in_channels = self.data_in[0], out_channels = 16,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(in_channels = 16, out_channels = 64, 
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels = self.data_in[0], kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            )         

        # First Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=self.data_in[0], out_channels=8, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=8)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, self.pool_size[0]))
        self.pool1avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[0]))
        
        # Second Convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, self.pool_size[1]))
        self.pool2avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[1]))

        # Third Convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=self.nb_cnn2d_filt, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=self.nb_cnn2d_filt)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, self.pool_size[2]))
        self.pool3avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[2]))

        # GRUS 2, 1, 0
        if self.n_grus == 2:
            self.gru1 = nn.GRU(input_size=int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), hidden_size=self.rnn_size[0], bidirectional=True, batch_first = True)
            self.gru2 = nn.GRU(input_size=self.rnn_size[0]*2, hidden_size=self.rnn_size[1], bidirectional=True, batch_first = True)
        elif self.n_grus == 1:
            self.gru1 = nn.GRU(input_size=int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), hidden_size=self.rnn_size[1], bidirectional=True, batch_first = True)
        elif self.n_grus == 0:
            self.gru_linear1 = nn.Linear(in_features = int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), out_features = self.rnn_size[0])
            self.gru_linear2 = nn.Linear(in_features=self.rnn_size[0], out_features=self.rnn_size[1]*2)
        else:
            raise ValueError

        self.fc1 = nn.Linear(in_features=self.rnn_size[1]*2, out_features=self.fnn_size)
        self.fc2 = nn.Linear(in_features=self.fnn_size, out_features = 1)

        self.final = nn.Linear(in_features = self.data_in[-2], out_features = 1)

    def normalize_tensor(self, x):
        mean = x.mean(dim = (2,3), keepdim = True)
        std = x.std(dim = (2,3), unbiased = False, keepdim = True)
        #std = torch.clamp(x.std(dim=(2, 3), unbiased=False, keepdim=True), min=1e-2)  
        return torch.div((x - mean), std ) 
    
    def normalize_tensor_zero_one(self, x): # risolve la comparsa di valori NaN solo per param_sinc
        min_val = x.min()
        max_val = x.max()

        if max_val == min_val:
            return torch.zeros_like(x)
        else:
            normalized_x = (x - min_val) / (max_val - min_val)
            return normalized_x

    def normalize_energy(self, x):
        energy = torch.sum(x ** 2)
        # Normalizza ciascun segnale dividendo per la radice quadrata dell'energia
        normalized_tensor = x / (torch.sqrt(energy) + 1e-9)
        return normalized_tensor
        
    def normalize_tensor_tanh(self, x):
        return torch.tanh(x)

    def forward(self, x):
        
        print(f"stampa magn: {x}") 
        print(f"stampa magn: {x.shape}")  #[16, 160000]

        x = x.unsqueeze(0)
        x = x.permute(1, 0, 2)


        # apply encoder
        enc_out = self.encoder(x)

        if self.encoder_type == 'stft':
            magn = transforms.mag(enc_out)
            magn = torch.log(magn**2 + 1e-7)
            magn = magn.unsqueeze(1)
            magn = magn.permute(0, 1, 3, 2)
            previous_magn = magn

            angles = transforms.angle(enc_out)
            angles = angles.unsqueeze(1)
            angles = angles.permute(0, 1, 3, 2)
            angles_cos = torch.cos(angles)
            angles_sin = torch.sin(angles)

            magn = magn[:,:,:,:-1]
            angles_cos = angles_cos[:,:,:,:-1]
            angles_sin = angles_sin[:,:,:,:-1]
            #print(f"dimensioni magn: {magn.shape}" )
           
            x = torch.cat((magn, angles_cos, angles_sin), dim = 1)
            #print(f"dimensioni tensore: {x.shape}" )

        elif self.encoder_type == 'analytic_free':
            #enc_out = torch.tanh(enc_out / torch.max(enc_out.abs()))
            enc_out = enc_out + 1e-3
            #normalized_out = self.normalize_tensor_zero_one(enc_out)

            magn = transforms.mag(enc_out)
            magn = torch.log(magn**2 + 1e-7)
            magn = magn.unsqueeze(1)
            magn = magn.permute(0, 1, 3, 2)
            previous_magn = magn
            print(f"stampa magn: {magn}")

            angles = transforms.angle(enc_out)
            angles = angles.unsqueeze(1)
            angles = angles.permute(0, 1, 3, 2)
            angles_cos = torch.cos(angles)
            angles_sin = torch.sin(angles)

            x = torch.cat((magn, angles_cos, angles_sin), dim = 1)

        elif self.encoder_type == 'free':
            #enc_out = torch.tanh(enc_out / torch.max(enc_out.abs()))
            enc_out = enc_out + 1e-9
            #normalized_out = self.normalize_tensor_zero_one(enc_out)
            #normalized_out = self.normalize_tensor_tanh(enc_out)

            magn = transforms.mag(enc_out)
            magn = torch.log(magn**2 + 1e-7)
            magn = magn.unsqueeze(1)
            magn = magn.permute(0, 1, 3, 2)
            previous_magn = magn
            print(f"stampa magn: {magn}")

            angles = transforms.angle(enc_out)
            angles = angles.unsqueeze(1)
            angles = angles.permute(0, 1, 3, 2)
            angles_cos = torch.cos(angles)
            angles_sin = torch.sin(angles)

            x = torch.cat((magn, angles_cos, angles_sin), dim = 1)

        elif self.encoder_type == "param_sinc":
            #enc_out = torch.tanh(enc_out / torch.max(enc_out.abs()))
            #enc_out = enc_out + 1e-3  #fondamentale per NaN
            #print(f"stampa pre norm: {enc_out.shape}")

            normalized_out = self.normalize_tensor_zero_one(enc_out) #basta da sola per evitare NaN
            #print(f"stampa post norm: {normalized_out.shape}")


            magn = transforms.mag(normalized_out)
            magn = torch.log(magn**2 + 1e-7)
            magn = magn.unsqueeze(1)
            magn = magn.permute(0, 1, 3, 2)
            previous_magn = magn
            print(f"stampa magn: {magn}")

            angles = transforms.angle(normalized_out)
            angles = angles.unsqueeze(1)
            angles = angles.permute(0, 1, 3, 2)
            angles_cos = torch.cos(angles)
            angles_sin = torch.sin(angles)

            x = torch.cat((magn, angles_cos, angles_sin), dim = 1)
        
        elif self.encoder_type == "custom_free":

           
            #enc_out = self.encoder(x)

            print(f"stampa magn: {enc_out}")


            magn = transforms.mag(enc_out)
            magn = torch.log(magn**2 + 1e-7)
            magn = magn.unsqueeze(1)
            magn = magn.permute(0, 1, 3, 2)
            previous_magn = magn
            print(f"stampa magn: {magn.shape}")

            angles = transforms.angle(enc_out)
            angles = angles.unsqueeze(1)
            angles = angles.permute(0, 1, 3, 2)
            angles_cos = torch.cos(angles)
            angles_sin = torch.sin(angles)

            x = torch.cat((magn, angles_cos, angles_sin), dim = 1)
        
            



                    
        x = self.normalize_tensor(x)

        # computation of the heatmap
        hm = self.heatmap(x)
        x = x * hm

        # convolutional layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.pool1(x) + self.pool1avg(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.pool2(x) + self.pool2avg(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.elu(x)
        x = self.pool3(x) + self.pool3avg(x)

        # recurrent layers (if any)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        if self.n_grus == 2:
            x, _ = self.gru1(x)
            x, _ = self.gru2(x)
        elif self.n_grus == 1:
            x, _ = self.gru1(x)
        else:
            x = self.gru_linear1(x)
            x = self.gru_linear2(x)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        
        x = x.squeeze(2) # here [batch_size, time_bins]
        rnn = x      
        x = self.final(x).squeeze()
        if self.att_conf == "Nothing":
            return x, rnn, previous_magn.detach(), None
        else:
            return x, rnn, previous_magn.detach(), hm.detach()
    

######################## Lightning Module

class SeldTrainer(LightningModule):
    def __init__(self, lr, kernels, n_grus, features_set, att_conf, n_filters, kernel_size, encoder_type, sampling_rate):
        super().__init__()

        # Hyperparameters
        self.sampling_rate = sampling_rate
        self.kernels = kernels
        self.n_grus = n_grus
        self.features_set = features_set
        self.att_conf = att_conf
        self.lr = lr
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.encoder_type = encoder_type
        self.evaluate = torch.nn.L1Loss()
        self.loss = torch.nn.MSELoss()
        self.model = SeldNet(self.kernels, self.n_grus, self.features_set, self.att_conf, self.n_filters, self.kernel_size, self.encoder_type, self.sampling_rate)
        self.all_test_results = []

    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        audio, labels, _ = batch['audio'], batch['label'], batch['id']
        distance_est, time_wise_distance, _ , _ = self(audio)
        loss = self.loss(distance_est, labels)
        loss_timewise = self.loss(torch.mean(time_wise_distance, dim = -1), labels)
        final_loss = (loss + loss_timewise)/2
        self.log("train/loss", loss, prog_bar = True, on_epoch = True, on_step = False)
        self.log("train/loss_timewise", loss_timewise, prog_bar = True, on_epoch = True, on_step = False)
        self.log("train/mae", self.evaluate(distance_est, labels), on_epoch = True, prog_bar = True, on_step = False)
        return final_loss

    def validation_step(self, batch, batch_idx):
        audio, labels, _ = batch['audio'], batch['label'], batch['id']
        distance_est, time_wise_distance, _, _= self(audio)
        loss = self.loss(distance_est, labels)
        loss_timewise = self.loss(torch.mean(time_wise_distance, dim = -1), labels)
        final_loss = (loss + loss_timewise)/2
        self.log("val/loss", loss, prog_bar = True, on_epoch = True)
        self.log("val/loss_timewise", loss_timewise, prog_bar = True, on_epoch = True, on_step = False)
        self.log("val/mae", self.evaluate(distance_est, labels), on_epoch = True, prog_bar = True)
        return final_loss
    
    def test_step(self, batch, batch_idx):
        audio, labels, ids = batch['audio'], batch['label'], batch['id']
        distance_est, time_wise_distance, _ ,_ = self(audio)
        loss = self.loss(distance_est, labels)
        loss_timewise = self.loss(torch.mean(time_wise_distance, dim = -1), labels)
        self.log("test/mae_overall", self.evaluate(distance_est, labels), on_epoch = True)
        # save everything
        for element in range(labels.shape[0]):
            data = {
                    'GT': labels[element].cpu().numpy(),
                    'Pred': distance_est[element].cpu().numpy(),
                    'L1': torch.abs(distance_est[element] - labels[element]).cpu().numpy(),
                    'rL1': (torch.abs(distance_est[element] - labels[element])/labels[element]).cpu().numpy(),
                    'ID': ids[element]
                }
            self.all_test_results.append(data)
        final_loss = (loss + loss_timewise)/2
        return final_loss
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr)
        return {
           "optimizer": opt,
           "lr_scheduler": {
               "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience = 5, factor = 0.2),
               "monitor": "val/loss",
               "frequency": 1
                           },
              }