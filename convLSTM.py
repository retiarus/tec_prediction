"""
The code is an adaptation from:

https://github.com/ndrplz/ConvLSTM_pytorch/
Copyright (c) 2017 Andrea Palazzi
under MIT License

to be used with dilations

The modfications are subject to the license from:
https://github.com/aboulch/tec_prediction
LGPLv3 for research and for commercial use see the LICENSE.md file in the repository
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

torch.set_default_dtype(torch.float32)

def swish(x):
    return x * torch.sigmoid(x)

class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.

    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size,
                 dilation=1,
                 act_cuda=False,
                 padding=None):
        """Init."""
        super(CLSTM_cell, self).__init__()
        self.act_cuda = act_cuda

        if padding is None:
            padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                              out_channels=4 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              stride=1,
                              padding=padding,
                              dilation=(dilation, dilation))

    def forward(self, input, prev_state=None):
        """Forward."""
        batch_size = input.data.size()[0]
        spatial_size = input.data.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if self.act_cuda:
                prev_state = [
                    Variable(torch.zeros(state_size)).cuda(),
                    Variable(torch.zeros(state_size)).cuda()
                ]
            else:
                prev_state = [
                    Variable(torch.zeros(state_size)),
                    Variable(torch.zeros(state_size))
                ]

        hidden, c = prev_state  # hidden and c are images with several channels
        combined = torch.cat((input, hidden), 1)  # oncatenate in the channels
        # print('combined',combined.size())
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.hidden_size,
                                       dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        #g = torch.tanh(ag) + swish(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        #next_h = o * (torch.tanh(next_c) + swish(next_c))
        return next_h, next_c
