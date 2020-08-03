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
from torch.nn import Conv1d, Linear, MaxPool1d, Module, ReLU, Sequential
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

class Dropout1d(nn.Module):
    def __init_(self, p: float = 0.2):
        super(ConvNet, self).__init__()
        self.drop = torch.nn.Dropout2d(p=p)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2), 1)
        x = self.drop(x)
        x = x.view(x.size(0), x.size(1), x.size(2))
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        num_features = 256
        self.conv1 = Conv1d(in_channels=1, out_channels=num_features, kernel_size=3, padding=1)
        self.conv2 = Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.conv3 = Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.conv4 = Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.conv5 = Conv1d(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1)
        self.conv6 = Conv1d(in_channels=num_features, out_channels=1, kernel_size=3, padding=1)

#        self.dropout1 = Dropout1d()
#        self.dropout2 = Dropout1d()
#        self.dropout3 = Dropout1d()

        self.norm1 = torch.nn.BatchNorm1d(num_features=num_features)
        self.norm2 = torch.nn.BatchNorm1d(num_features=num_features)
        self.norm3 = torch.nn.BatchNorm1d(num_features=num_features)
        self.norm4 = torch.nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        x = x[:, :, -48:]
        x1 = torch.relu(self.conv1(x))
        x1 = self.norm1(x1)
 #       x1 = self.dropout1(x1)
        x2 = torch.relu(self.conv2(x1))
        x2 = self.norm2(x2)
#        x2 = self.dropout2(x2)
        x3 = torch.relu(self.conv3(x2)) + x1
        x3 = self.norm2(x3)
#        x3 = self.dropout3(x3)
        x4 = torch.relu(self.conv4(x3))
        x4 = self.norm3(x4)
        x5 = torch.relu(self.conv5(x4)) + x2
        x6 = self.conv6(x5)

        return x6

model = ConvNet()

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

    torch.save(model.state_dict(), './simple_model.pth')
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
