"""
Simple network

License is from https://github.com/aboulch/tec_prediction
"""

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from convLSTM import CLSTM_cell as Recurrent_cell

torch.set_default_dtype(torch.float32)

def swish(x):
    return x * torch.sigmoid(x)

class SimpleConvRecurrent(nn.Module):
    """Segnet network."""
    def __init__(self, input_nbr, num_features=128, act_cuda=False):
        """Init fields."""
        super(SimpleConvRecurrent, self).__init__()

        self.input_nbr = input_nbr
        self.act_cuda = act_cuda
        self.name = 'simple_conv_recurrent_net'

        self.init_encoder()
        self.convRecurrentCell = Recurrent_cell(num_features,
                                                num_features,
                                                kernel_size=3,
                                                act_cuda=self.act_cuda)
        self.init_decoder()

        self.init_h()
        self.init_hidden()


    def forward(self, z, prediction_len,
                diff=False, predict_diff_data=None):
        """Forward method."""
        output_inner = []
        size = z.size()
        seq_len = z.size(1)

        h = self.conv_2d(z)

        hidden_state = None
        hidden_rc = []
        for t in range(seq_len):  #loop for every step
            x = h[:, t, ...]

            hidden_state = self.convRecurrentCell(x, hidden_state)

            y = hidden_state[0]
            if t > seq_len - seq_predict:
                hidden_inner.append(y)

            y = self.decoder(y)

        output_inner.append(y)

        hidden_rc = torch.stack(hidden_rc, 1)
        hidden_rc_short = self.hidden_short(hidden_inner)
#        y.repeat((1, seq_len))

        for t in range(prediction_len - 1):  #loop for every step
            x = y.view(y.size(0), y.size(1), self.input_nbr, y.size(2), y.size(3))
            x = self.conv_2d(x)
            x = x.view(x.size(0), x.size(2), x.size(3), x.size(4))

            hidden_state = self.convRecurrentCell(x, hidden_state)

            y = hidden_state[0]
            y = self.decoder(y)

            output_inner.append(y)

        ouput = torch.stack(output_inner, 1) + hidden_inner_short

        return output

    def conv_2d(self, z):
        x = z.transpose(2, 1)
#        x = self.batch_norm(x) + self.preserved*x
#        x = gaussian(x, is_training=True, stddev=0.1)

        x = swish(self.conv1(x))
#        x = self.batch_norm1(x)
#        x = gaussian(x, is_training=True, stddev=0.1)

        x = swish(self.conv2(x))
#        x = self.batch_norm2(x)
#        x = gaussian(x, is_training=True, stddev=0.1)

        x = swish(self.conv3(x))
#        x = self.batch_norm3(x)
#        x = gaussian(x, is_training=True, stddev=0.1)

        x = swish(self.conv4(x))
 #       x = self.batch_norm4(x)

        return x.transpose(2, 1)

    def load_from_filename(self, model_path):
        """Load weights method."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)

    def init_hidden(self):
        self.convd_hidden_4 = nn.ConvTranspose3d(num_features,
                                          num_features,
                                          kernel_size=(3, 3, 3),
                                          padding=(1, 1, 1),
                                          stride=(1, 2, 2),
                                          output_padding=(0, 1, 1))
        self.convd_hidden_3 = nn.ConvTranspose3d(num_features,
                                         num_features,
                                         kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1),
                                         stride=(1, 2, 2),
                                         output_padding=(0, 1, 1))
        self.convd_hidden_2 = nn.ConvTranspose3d(num_features,
                                         num_features,
                                         kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1),
                                         stride=(1, 2, 2),
                                         output_padding=(0, 1, 1))
        self.convd_hidden_1 = nn.ConvTranspose3d(num_features,
                                         input_nbr,
                                         kernel_size=3,
                                         padding=1,
                                         stride=2,
                                         output_padding=1)

    def hidden_short(self, hidden):
        pdb.set_trace()
        hidden = swish(self.convd_hidden_4(hidden))
        hidden = swish(self.convd_hidden_3(hidden))
        hidden = swish(self.convd_hidden_2(hidden))
        hidden = self.convd_hidden_1(hidden)

        return hidden

    def init_h(self):
        self.convd_h_4 = nn.ConvTranspose3d(num_features,
                                          num_features,
                                          kernel_size=(3, 3, 3),
                                          padding=(1, 1, 1),
                                          stride=(1, 2, 2),
                                          output_padding=(0, 1, 1))
        self.convd_h_3 = nn.ConvTranspose3d(num_features,
                                         num_features,
                                         kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1),
                                         stride=(1, 2, ,2),
                                         output_padding=1)
        self.convd_h_2 = nn.ConvTranspose3d(num_features,
                                         num_features,
                                         kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1),
                                         stride=(1, 2, 2),
                                         output_padding=1)
        self.convd_h_1 = nn.ConvTranspose3d(num_features,
                                         input_nbr,
                                         kernel_size=(3, 3, 3),
                                         padding=(1, 1, 1),
                                         stride=(1, 2, 2),
                                         output_padding=1)
    def init_decoder(self):
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

    def init_encoder(self):
        self.conv1 = nn.Conv3d(input_nbr,
                               num_features,
                               kernel_size=(3, 3, 3),
                               padding=(1, 1, 1),
                               stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(num_features,
                               num_features,
                               kernel_size=(3, 3, 3),
                               padding=(1, 1, 1),
                               stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(num_features,
                               num_features,
                               kernel_size=(3, 3, 3),
                               padding=(1, 1, 1),
                               stride=(1, 2, 2))
        self.conv4 = nn.Conv3d(num_features,
                               num_features,
                               kernel_size=1,
                               padding=0)

#        self.preserved = torch.autograd.Variable(torch.ones(1), requires_grad=True)
#        self.batch_norm = nn.BatchNorm3d(input_nbr)
#        self.batch_norm1 = nn.BatchNorm3d(num_features)
#        self.batch_norm2 = nn.BatchNorm3d(num_features)
#        self.batch_norm3 = nn.BatchNorm3d(num_features)
#        self.batch_norm4 = nn.BatchNorm3d(num_features)

    def decoder(self, x):
        x = swish(self.convd4(x))
        x = swish(self.convd3(x))
        x = swish(self.convd2(x))
        return self.convd1(x)


def gaussian(ins, is_training=False, mean=0.0, stddev=0.1):
    if is_training:
        noise = torch.autograd.Variable(ins.data.new(ins.size()).normal_(mean, stddev), requires_grad=False)
        return ins + noise

    return ins
