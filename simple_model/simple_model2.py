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


class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.hidden_size = hidden_size = 16
        self.len_train = len_tain = 96
        self.len_predict = len_predict = 48
        self.num_layers = 24

        self.lstm1 = LSTM(input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=True)

        self.lstm2 = LSTM(input_size=2*self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=True)

        self.fc1 = Linear(2*self.len_predict*self.hidden_size, 1000)
        self.fc2 = Linear(1000, self.len_predict)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x1 = x[:, -48:, :]

        x, _ = self.lstm2(x1)
        x = x + x1
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(x.size(0), self.len_predict, 1)
        x = x.permute(0, 2, 1)
        return x

model = Lstm()

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

    torch.save(model.state_dict(), './simple_model2.pth')
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
