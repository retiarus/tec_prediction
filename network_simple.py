"""
Simple network

License is from https://github.com/aboulch/tec_prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from convLSTM import CLSTM_cell as Recurrent_cell

import pdb
torch.set_default_dtype(torch.float32)


class SimpleConvRecurrent(nn.Module):
    """Segnet network."""
    def __init__(self, input_nbr, num_features=8, act_cuda=False):
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
        self.conv3 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size=3,
                               padding=1,
                               stride=2)
        self.conv4 = nn.Conv2d(num_features,
                               num_features,
                               kernel_size=1,
                               padding=0)

        kernel_size = 3
        self.convRecurrentCell = Recurrent_cell(num_features,
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

        output_inner = []
        size = z.size()
        seq_len = z.size(1)
        # hidden_state=self.convLSTM1.init_hidden(size[1])
        hidden_state = None
        for t in range(seq_len):  #loop for every step
            x = z[:, t, ...]

            # coder
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))

            # recurrent
            hidden_state = self.convRecurrentCell(x, hidden_state)

            y = hidden_state[0]

            y = F.relu(self.convd4(y))
            y = F.relu(self.convd3(y))
            y = F.relu(self.convd2(y))
            y = self.convd1(y)

        output_inner.append(y)

        for t in range(prediction_len - 1):  #loop for every step

            if (diff):
                x = y + predict_diff_data[:, t, ...]
            else:
                x = y

            # coder
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))

            # recurrent
            hidden_state = self.convRecurrentCell(x, hidden_state)

            y = hidden_state[0]

            y = F.relu(self.convd4(y))
            y = F.relu(self.convd3(y))
            y = F.relu(self.convd2(y))
            y = self.convd1(y)

            output_inner.append(y)

        current_input = torch.stack(output_inner, 1)

        return current_input

    def load_from_filename(self, model_path):
        """Load weights method."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)
