import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from convLSTM import CLSTM_cell as Recurrent_cell
from torch.autograd import Variable

torch.set_default_dtype(torch.float32)


def swish(x):
    return x * torch.sigmoid(x)


class SimpleConvRecurrent(nn.Module):
    def __init__(self, input_nbr, num_features=128, act_cuda=False):
        """Init fields."""
        super(SimpleConvRecurrent, self).__init__()

        self.input_nbr = input_nbr
        self.num_features = num_features
        self.act_cuda = act_cuda
        self.name = 'simple_conv_recurrent_net_1'

#        self.conv1 = nn.Conv3d(input_nbr,
#                               num_features,
#                               kernel_size=(3, 3, 3),
#                               padding=1,
#                               stride=1)

#        self.conv2 = nn.Conv3d(num_features,
#                               num_features,
#                               kernel_size=(3, 3, 3),
#                               padding=1,
#                               stride=1)

#        self.conv3 = nn.Conv3d(num_features,
#                               num_features,
#                               kernel_size=(3, 3, 3),
#                               padding=1,
#                               stride=1)

        kernel_size = 3
        self.convRecCell1 = Recurrent_cell(input_nbr,
                                           num_features,
                                                 kernel_size,
                                                 act_cuda=self.act_cuda)

        self.convRecCell2 = Recurrent_cell(num_features,
                                                 num_features,
                                                 kernel_size,
                                                 act_cuda=self.act_cuda)

        self.convRecCell3 = Recurrent_cell(num_features,
                                                 num_features,
                                                 kernel_size,
                                                 act_cuda=self.act_cuda)
        self.convRecCell4 = Recurrent_cell(num_features,
                                                 num_features,
                                                 kernel_size,
                                                 act_cuda=self.act_cuda)

        self.conv4 = nn.Conv3d(num_features,
                               num_features,
                               kernel_size=(3, 3, 3),
                               padding=1,
                               stride=1)

#        self.conv5 = nn.Conv3d(num_features,
#                               num_features,
#                               kernel_size=(3, 3, 3),
#                               padding=1,
#                               stride=1)

        self.conv6 = nn.Conv3d(num_features,
                               input_nbr,
                               kernel_size=(3, 3, 3),
                               padding=1,
                               stride=1)

        #self.fc1 = nn.Linear(num_features*48, 10000)
        #self.fc2 = nn.Linear(10000, 48)

        self.batch_norm1 = nn.BatchNorm3d(num_features)
        self.batch_norm2 = nn.BatchNorm3d(num_features)
#        self.batch_norm3 = nn.BatchNorm3d(num_features)
        self.batch_norm4 = nn.BatchNorm3d(num_features)
#        self.batch_norm5 = nn.BatchNorm3d(num_features)
        self.batch_norm_x3 = nn.BatchNorm3d(num_features)
        self.batch_norm_x4 = nn.BatchNorm3d(num_features)

    def forward(self, z, prediction_len,
                diff=False, predict_diff_data=None):
        """Forward method."""
        size = z.size()
        seq_len = z.size(1)

#        x = z.transpose(2, 1)

#        x1 = swish(self.conv1(x))
 #       x1 = self.batch_norm1(x1+x.repeat(1, 96, 1, 1, 1))

#        x2 = swish(self.conv2(x1))
#        x2 = self.batch_norm2(x2+x.repeat(1, 96, 1, 1, 1))

#        x = swish(self.conv3(x))
#        x = self.batch_norm3(x)

 #       x = x.transpose(2, 1)


        hidden_state1 = None
        hidden_state2 = None
        hidden_state3 = None
        hidden_state4 = None
        output_rec3 = []
        output_rec4 = []
#        z = x
        for t in range(seq_len):
            x = z[:, t, ...]
            hidden_state1 = self.convRecCell1(x, hidden_state1)
            hidden_state2 = self.convRecCell2(hidden_state1[0], hidden_state2)
            hidden_state3 = self.convRecCell3((hidden_state2[0] + hidden_state1[0])/2.0, hidden_state3)
            hidden_state4 = self.convRecCell4((hidden_state3[0] + hidden_state2[0])/2.0, hidden_state4)

            if t >= seq_len - prediction_len:
                output_rec3.append(hidden_state3[0])
                output_rec4.append(hidden_state4[0])

        x3 = torch.stack(output_rec3, 1)
        x3 = x3.transpose(2, 1)
        x3 = self.batch_norm_x3(x3)

        x4 = torch.stack(output_rec4, 1)
        x4 = x4.transpose(2, 1)
        x4 = self.batch_norm_x4(x4)

        x = swish(self.conv4(x4))
        x = self.batch_norm4(x)

        x = x + x3

#        x = swish(self.conv5(x))
#        x = self.batch_norm5(x)

        x = self.conv6(x)

        x = x.transpose(2, 1)
#        x = x.transpose(1, 3)
#        x = x.transpose(2, 4)
#        size = x.size
#        x = x.view(-1, prediction_len*self.num_features)
#        x = self.fc1(x)
#        x = self.fc2(x)
#        x = x.view(size(0), size(1), size(2), prediction_len, self.input_nbr)
#        x = x.transpose(1, 3)
#        x = x.transpose(2, 4)

        return x

    def load_from_filename(self, model_path):
        """Load weights method."""
        th = torch.load(model_path, map_location=torch.device('cpu'))  # load the weigths
        self.load_state_dict(th)
