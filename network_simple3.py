"""
Simple network

License is from https://github.com/aboulch/tec_prediction
"""

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from convLSTM import CLSTM_cell as Recurrent_cell

torch.set_default_dtype(torch.float32)

def swish(x):
    return x * torch.sigmoid(x)

def CombinedConv(nn.Module):
    def __init__(self, input_nbr, num_features, act_cuda=False):
        super(CombinedConv, self).__init__()
        self.conv_3x3 = nn.Conv2d(input_nbr,
                                  num_features,
                                  kernel_size=3,
                                  padding=2,
                                  stride=1)
        self.conv_5x5 = nn.Conv2d(input_nbr,
                                  num_features,
                                  kernel_size=5,
                                  padding=4,
                                  stride=1)
        self.conv_7x7 = nn.Conv2d(input_nbr,
                                  num_features,
                                  kernel_size=7,
                                  padding=6,
                                  stride=1)

    def forward(x):
        x_3x3 = swish(self.conv_3x3(x))
        x_5x5 = swish(self.conv_5x5(x))
        x_7x7 = swish(self.conv_7x7(x))
        #combined = torch.cat((input, hidden), 1)  # oncatenate in the channels
        #current_input = torch.stack(output_inner, 1)



class SimpleConvRecurrent(nn.Module):
    """Segnet network."""
    def __init__(self, input_nbr, num_features=32, act_cuda=False):
        """Init fields."""
        super(SimpleConvRecurrent, self).__init__()

        self.act_cuda = act_cuda
        self.name = 'simple_conv_recurrent_net'

        self.conv1 = nn.Conv2d(input_nbr,
                               num_features,
                               kernel_size=3,
                               padding=1,
                               stride=2)
        self.conv2 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size=3,
                               padding=1,
                               stride=2)
        self.preserved1 = Variable(torch.ones(1), requires_grad=True)
        self.layer_norm1 = torch.nn.LayerNorm()
        self.conv3 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size=3,
                               padding=1,
                               stride=2)
        self.conv4 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size=1,
                               padding=0)
        self.conv5_ramo_1 = nn.Conv3d(num_features,
                                      num_features,
                                      kernel_size=1,
                                      padding=2)
        self.conv5_ramo_2 = nn.Conv3d(num_features,
                                      num_features,
                                      kernel_size=1,
                                      padding=2)

        self.preserved_recurrent = Variable(torch.ones(1), requires_grad=True)
        self.layer_recurrent = torch.nn.LayerNorm()
        
        self.preserved2 = Variable(torch.ones(1), requires_grad=True)
        self.layer_norm2 = torch.nn.LayerNorm()
        
        self.preserved3 = Variable(torch.ones(1), requires_grad=True)
        self.layer_norm3 = torch.nn.LayerNorm()

        kernel_size = 3
        self.convRecurrentCell1 = Recurrent_cell(num_features,
                                                 num_features,
                                                 kernel_size,
                                                 act_cuda=self.act_cuda)
        
        kernel_size = 3
        self.convRecurrentCell2 = Recurrent_cell(num_features,
                                                 num_features,
                                                 kernel_size,
                                                 act_cuda=self.act_cuda)

        self.convd4 = nn.Conv2d(num_features,
                                num_features,
                                kernel_size=1,
                                padding=0)
        self.convd3 = nn.ConvTranspose2d(num_features,
                                         num_features,
                                         kernel_size=3,
                                         padding=1,
                                         stride=2,
                                         output_padding=1)
        self.convd2 = nn.ConvTranspose2d(num_features,
                                         num_features,
                                         kernel_size=3,
                                         padding=1,
                                         stride=2,
                                         output_padding=1)
        self.convd1 = nn.ConvTranspose2d(num_features,
                                         input_nbr,
                                         kernel_size=3,
                                         padding=1,
                                         stride=2,
                                         output_padding=1)

    def forward(self, z, prediction_len,
                diff=False, predict_diff_data=None):
        """Forward method."""

#        z = self.layer_norm1(z) + self.preserved1(z)

        output_inner = []
        size = z.size()
        seq_len = z.size(1)
        # hidden_state=self.convLSTM1.init_hidden(size[1])
        hidden_state = None

        z = z.view(-1, z.size(2), z.size(3), z.size(4))
        z = swish(self.conv1(x))
        z = swish(self.conv2(x))
        z = swish(self.conv3(x))
        z = swish(self.conv4(x))
        z = x.view(size[0], size[1], x.size(1), x.size(2), z.size(3))
        z = x.transpose(0, 2, 1, 3, 4)
        z = self.preservedcurrent*x + self.layer_recurrent(x)

        for t in range(seq_len):  #loop for every step
            x = z[:, t, ...]

            # recurrent
            hidden_state1 = self.convRecurrentCell(x, hidden_state)
            hidden_state2 = self.convRecurrentCell(x, hidden_state)

            y1 = hidden_state[0]
            y2 = hidden_state[0]

            y1 = swish(self.convd4(y))
            y1 = swish(self.convd3(y))
            y1 = swish(self.convd2(y))
            y1 = self.convd1(y)
            
            y2 = swish(self.convd4(y))
            y2 = swish(self.convd3(y))
            y2 = swish(self.convd2(y))
            y2 = self.convd1(y)
           
           if t > seq_len - prediction_len:
               output_inner.append(y1)

        y1 = torch.stack(output_inner, 1)
        y2 = y2.repeat()

        y = swish(self.conv5_ramo_1(y1)) + swish(self.conv5_ramo_1(y1))

        return current_input

    def load_from_filename(self, model_path):
        """Load weights method."""
        th = torch.load(model_path, map_location=torch.device('cpu'))  # load the weigths
        self.load_state_dict(th)


def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise
    return ins
