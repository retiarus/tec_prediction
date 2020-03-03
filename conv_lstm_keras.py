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

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ZeroPadding2D

tf.keras.backend.set_floatx('float32')


class CLSTM_cell(Model):
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
                 padding=None):
        """Init."""
        super(CLSTM_cell, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.padding = ZeroPadding2D(padding=(padding, padding),
                                     data_format='channels_first')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = Conv2D(4 * self.hidden_size,
                           self.kernel_size,
                           strides=(1, 1),
                           padding='valid',
                           dilation_rate=dilation,
                           data_format='channels_first')

    @tf.function
    def call(self, input_tensor, prev_state=None, training=False):
        """Forward."""
        batch_size = tf.shape(input_tensor).numpy()[0]
        spatial_size = tf.shape(input_tensor).numpy()[2:]

        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = [tf.zeros(state_size), tf.zeros(state_size)]

        hidden, c = prev_state  # hidden and c are images with several channels
        combined = tf.concat([input_tensor, hidden],
                             1)  # oncatenate in the channels
        # print('combined',combined.size())
        combined = self.padding(combined)
        A = self.conv(combined)
        (ai, af, ao, ag) = tf.split(A, num_or_size_splits=4,
                                    axis=1)  # it should return 4 tensors
        i = tf.math.sigmoid(ai)
        f = tf.math.sigmoid(af)
        o = tf.math.sigmoid(ao)
        g = tf.nn.relu(ag)

        next_c = f * c + i * g
        next_h = o * tf.nn.relu(next_c)
        return next_h, next_c
