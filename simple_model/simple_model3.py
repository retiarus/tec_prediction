import os
import pdb

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import LSTM, Conv1d, Linear, MaxPool1d, Module, ReLU, Sequential
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader_torch2 import SequenceLoader

tb = SummaryWriter()

model = "simple"
tool = "pytorch"
data = "scin"
station = "summer"
seq_length_min = int(3*60*24)
step_min = 30
window_train = int(2*60*24/step_min)
window_predict = int(1*60*24/step_min)
batch_size = 50
to_fit = False

print(f"Sequence length: {seq_length_min}")
print(f"Step: {step_min}")
print(f"Window train: {window_train}")
print(f"Window predict: {window_predict}")

ds_train = SequenceLoader(name='train',
                          path_files='/mnt/data/resized_type_1',
                          seq_length_min=seq_length_min,
                          step_min=step_min,
                          window_train=window_train,
                          window_predict=window_predict,
                          data=data,
                          station=station)

seq_train = torch.utils.data.DataLoader(ds_train,
                                        batch_size=50,
                                        shuffle=True,
                                        num_workers=3,
                                        pin_memory=True)

ds_test = SequenceLoader(name='test',
                          path_files='/mnt/data/resized_type_1',
                          seq_length_min=seq_length_min,
                          step_min=step_min,
                          window_train=window_train,
                          window_predict=window_predict,
                          data=data,
                          station=station)

seq_test = torch.utils.data.DataLoader(ds_test,
                                        batch_size=50,
                                        shuffle=True,
                                        num_workers=2,
                                        pin_memory=True)

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.chomp1(out)
        out = self.dropout1(out)
        
        out = torch.relu(self.conv2(out))
        out = self.chomp2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return out + res
    
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN1(nn.Module):
    def __init__(self, n_inputs, n_outputs, x_len, y_len, num_in_layers=2):
        super(TCN1, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.x_len = x_len
        self.y_len = y_len
        self.num_channels = num_in_layers
        
        hidden_channels1 = [n_outputs for i in range(num_in_layers)]
        self.tcn1 = TemporalConvNet(num_inputs=1, num_channels=[self.n_inputs, *hidden_channels1])
        self.norm1 = nn.BatchNorm1d(n_outputs)
        
        hidden_channels2 = [n_outputs for i in range(num_in_layers)]
        self.tcn2 = TemporalConvNet(num_inputs=self.n_outputs, num_channels=[*hidden_channels2])
        self.norm2 = nn.BatchNorm1d(n_outputs)
        
        hidden_channels3 = [n_outputs for i in range(num_in_layers)]
        self.tcn3 = TemporalConvNet(num_inputs=self.n_outputs, num_channels=hidden_channels3)
        self.norm3 = nn.BatchNorm1d(n_outputs)
        
        hidden_channels4 = [n_outputs for i in range(num_in_layers)]
        self.tcn4 = TemporalConvNet(num_inputs=self.n_outputs, num_channels=[*hidden_channels4, self.n_inputs])
            
    def forward(self, x):
        x1 = self.tcn1(x)
        x2 = self.norm1(x1)
        
        x2 = self.tcn2(x1)
        x2 = x2[:, :, -self.y_len:].contiguous()
        x2 = self.norm1(x2)
        
        x3 = self.tcn3(x2)
        x3 = self.norm3(x2)
        z = self.tcn4(x3)
       
        return z

model = TCN1(n_inputs=1,
            n_outputs=128,
            x_len=96,
            y_len=48)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


list_loss_g_train = []
list_loss_g_test = []
num_epochs = 1000
t = tqdm(range(num_epochs), ncols=140)
for epoch in t:
    loss_train = []
    loss_test = []
    a = tqdm(seq_train, ncols=140)
    for seq in a:
        X, y = seq
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss_train.append(loss.item())

        loss.backward()
        optimizer.step()
        a.set_postfix(Loss=loss.detach().numpy())

    with torch.no_grad():
        for seq in seq_test:
            X, y = seq
            output = model(X)
            loss = criterion(output, y)
            loss_test.append(loss.item())

    torch.save(model.state_dict(), './simple_model3.pth')
    loss_g_train = np.mean(loss_train)
    list_loss_g_train.append(loss_g_train)
    loss_g_test = np.mean(loss_test)
    list_loss_g_test.append(loss_g_test)

    tb.add_scalars('loss',
                   {'train': loss_g_train,
                    'test': loss_g_test},
                   epoch)

#    print(f'Epoch [{epoch + 1}/{num_epochs}], loss_train: {loss_g_train:.4f}, loss_test: {loss_g_test:.4f}')
    t.set_postfix(LossTrain=loss_g_train,
                  LossTest=loss_g_test)
