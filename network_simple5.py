"""
Simple network

License is from https://github.com/aboulch/tec_prediction
"""

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from convlstm import ConvLSTM

torch.set_default_dtype(torch.float32)

class Encoder(nn.Module):
    def __init__(self, input_nbr, num_features=128, act_cuda=False):
        super(Encoder, self).__init__()
 
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

      def forward(self, x):
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


class Decoder(nn.Module):
    def __init__(self, input_nbr, num_features=128, act_cuda=False):
        super(Decoder, self).__init__()
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
        
    def forward(self, x):
        x = swish(self.convd4(x))
        x = swish(self.convd3(x))
        x = swish(self.convd2(x))
        return self.convd1(x)
 
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

      def forward(self, x):
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

class SimpleConvRecurrent(nn.Module):
    """Segnet network."""
    def __init__(self, input_nbr, num_features=128, act_cuda=False):
        """Init fields."""
        super(SimpleConvRecurrent, self).__init__()

        self.input_nbr = input_nbr
        self.act_cuda = act_cuda
        self.name = 'simple_conv_recurrent_net'

        hidden_dim = 8
        num_layers = 2

        self.encoder = Encoder()
        self.convlstm1 = ConvLSTM(input_dim=1,
                                  hidden_dim=hidden_dim,
                                  kernel_size=(3, 3),
                                  num_layers=num_layers,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)
        self.decoder = Decoder()

    def forward(self, z, prediction_len,
                diff=False, predict_diff_data=None):
        """Forward method."""
        output_inner = []
        size = z.size()

        x = self.encoder(z)
        x = self.convlstm1(x)
        x = self.decoder(x)

        y = self.encoder(z)
        y = self.decoder(y)
        return x, y

    def load_from_filename(self, model_path):
        """Load weights method."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)


def gaussian(ins, is_training=False, mean=0.0, stddev=0.1):
    if is_training:
        noise = torch.autograd.Variable(ins.data.new(ins.size()).normal_(mean, stddev), requires_grad=False)
        return ins + noise

    return ins
