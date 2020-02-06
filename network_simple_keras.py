from conv_lstm_keras import CLSTM_cell as RecurrentCell
from tf.keras import Model
from tf.keras.layers import Conv2D, Conv2DTranspose
from tf.nn import relu


class simple_conv_recurrent(Model):
    def __init__(self, input_nbr, num_map_features=8):
        super(simple_conv_recurrent, self).__init__()

        self.conv1 = Conv2D(num_map_features,
                            kernel_size=3,
                            padding=(1, 1),
                            strids=(2, 2),
                            data_format='channels_first')
        self.conv2 = Conv2D(num_map_features,
                            kernel_size=3,
                            padding=(1, 1),
                            stride=(2, 2),
                            data_format='channels_first')
        self.conv3 = Conv2D(num_map_features,
                            kernel_size=3,
                            padding=(1, 1),
                            stride=(2, 2),
                            data_format='channels_first')
        self.conv4 = Conv2D(num_map_features,
                            kernel_size=1,
                            padding=(0, 0),
                            data_format='channels_first')

        kernel_size = 3
        self.conv_recurrent_cell = RecurrentCell(num_map_features,
                                                 num_map_features, kernel_size)

        self.convd4 = Conv2D(num_map_features,
                             kernel_size=1,
                             padding=(0, 0),
                             data_format='channels_first')
        self.convd3 = Conv2DTranspose(num_map_features,
                                      kernel_size=3,
                                      padding=(1, 1),
                                      stride=(2, 2),
                                      output_padding=(1, 1))
        self.convd2 = Conv2DTranspose(num_map_features,
                                      kernel_size=3,
                                      padding=(1, 1),
                                      stride=(2, 2),
                                      output_padding=(1, 1))
        self.convd1 = Conv2DTranspose(input_nbr,
                                      kernel_size=3,
                                      padding=(1, 1),
                                      stride=(2, 2),
                                      output_padding=(1, 1))

        def call(self, z, window_predict, diff=False, predict_diff_data=None):
            output_inner = []
            size = z.shape
            seq_len = z.shape[0]

            hidden_state = None
            for t in range(seq_len):
                x = z[t, ...]
                x = relu(self.conv1(x))
                x = relu(self.conv2(x))
                x = relu(self.conv3(x))
                x = relu(self.conv4(x))

                hidden_state = self.conv_recurrent_cell(x, hidden_state)

                y = hidden_state[0]

                y = relu(self.convd4(y))
                y = relu(self.convd3(y))
                y = relu(self.convd2(y))
                y = self.convd1(y)

            output_inner.append(y)

            for t in range(window_predict - 1):  #loop for every step

                if (diff):
                    x = y + predict_diff_data[t, ...]
                else:
                    x = y

                # coder
                x = relu(self.conv1(x))
                x = relu(self.conv2(x))
                x = relu(self.conv3(x))
                x = relu(self.conv4(x))

                # recurrent
                hidden_state = self.convRecurrentCell(x, hidden_state)

                y = hidden_state[0]

                y = relu(self.convd4(y))
                y = relu(self.convd3(y))
                y = relu(self.convd2(y))
                y = self.convd1(y)

                output_inner.append(y)


#            expected_size = (len(output_inner), z.size(1), z.size(2),
#                             z.size(3), z.size(4))
#           current_input = torch.cat(output_inner, 0).view(expected_size)

#            return current_input
