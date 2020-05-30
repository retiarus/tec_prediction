"""
Dilated convolutional network

License is from https://github.com/aboulch/tec_prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from convLSTM import CLSTM_cell as Recurrent_cell


class DilationConvRecurrent(nn.Module):
    """Segnet network."""
    def __init__(self, input_nbr, num_features=8, act_cuda=False):
        """Init fields."""
        super(DilationConvRecurrent, self).__init__()

        self.act_cuda = act_cuda
        kernel_size = 3

        self.convRecurrentCell1 = Recurrent_cell(input_nbr,
                                                 num_features,
                                                 kernel_size,
                                                 dilation=1,
                                                 padding=1,
                                                 act_cuda=self.act_cuda)
        self.convRecurrentCell2 = Recurrent_cell(num_features,
                                                 num_features,
                                                 kernel_size,
                                                 dilation=2,
                                                 padding=2,
                                                 act_cuda=self.act_cuda)
        self.convRecurrentCell3 = Recurrent_cell(num_features,
                                                 input_nbr,
                                                 kernel_size,
                                                 dilation=1,
                                                 padding=1,
                                                 act_cuda=self.act_cuda)

    def forward(self, z, prediction_len, diff=False, predict_diff_data=None):
        """Forward method."""
        # Stage 1

        output_inner = []
        seq_len = z.size(1)

        hidden_state1 = None
        hidden_state2 = None
        hidden_state3 = None
        for t in range(seq_len):  # loop for every step
            x = z[:, t, ...]

            # coder
            x1 = x
            hidden_state1 = self.convRecurrentCell1(x1, hidden_state1)
            x1 = hidden_state1[0]
            x1 = F.relu(x1)

            x2 = x1
            hidden_state2 = self.convRecurrentCell2(x2, hidden_state2)
            x2 = hidden_state2[0]
            x2 = F.relu(x2)

            x3 = x2
            hidden_state3 = self.convRecurrentCell3(x3, hidden_state3)
            x3 = hidden_state3[0]

            y = x3

        output_inner.append(y)

        for t in range(prediction_len - 1):

            if (diff):
                x = y + predict_diff_data[:, t, ...]
            else:
                x = y

            # coder
            x1 = x
            hidden_state1 = self.convRecurrentCell1(x1, hidden_state1)
            x1 = hidden_state1[0]
            x1 = F.relu(x1)

            x2 = x1
            hidden_state2 = self.convRecurrentCell2(x2, hidden_state2)
            x2 = hidden_state2[0]
            x2 = F.relu(x2)

            x3 = x2
            hidden_state3 = self.convRecurrentCell3(x3, hidden_state3)
            x3 = hidden_state3[0]

            y = x3

            output_inner.append(y)

        current_input = torch.stack(output_inner, 1)

        return current_input

    def load_from_filename(self, model_path):
        """Load weights method."""
        th = torch.load(model_path)  # load the weigths
        self.load_state_dict(th)
