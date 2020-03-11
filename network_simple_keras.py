import tensorflow as tf
from conv_lstm_keras import CLSTM_cell as RecurrentCell
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D

tf.keras.backend.set_floatx('float32')


class SimpleConvRecurrent(Model):
    def __init__(self, input_nbr, num_map_features=16, diff=False):
        super(SimpleConvRecurrent, self).__init__()
        self.diff = False

        self.padding = ZeroPadding2D(padding=(1, 1),
                                     data_format='channels_first')
        self.conv1 = Conv2D(num_map_features,
                            kernel_size=3,
                            padding='valid',
                            strides=(2, 2),
                            data_format='channels_first')
        self.conv2 = Conv2D(num_map_features,
                            kernel_size=3,
                            padding='valid',
                            strides=(2, 2),
                            data_format='channels_first')
        self.conv3 = Conv2D(num_map_features,
                            kernel_size=3,
                            padding='valid',
                            strides=(2, 2),
                            data_format='channels_first')
        self.conv4 = Conv2D(num_map_features,
                            kernel_size=1,
                            padding='valid',
                            data_format='channels_first')

        kernel_size = 3
        self.conv_recurrent_cell = RecurrentCell(num_map_features,
                                                 num_map_features, kernel_size)

        self.convd4 = Conv2D(num_map_features,
                             kernel_size=1,
                             padding='same',
                             data_format='channels_first')
        self.convd3 = Conv2DTranspose(num_map_features,
                                      kernel_size=3,
                                      padding='same',
                                      strides=(2, 2),
                                      output_padding=(1, 1),
                                      data_format='channels_first')
        self.convd2 = Conv2DTranspose(num_map_features,
                                      kernel_size=3,
                                      padding='same',
                                      strides=(2, 2),
                                      output_padding=(1, 1),
                                      data_format='channels_first')
        self.convd1 = Conv2DTranspose(input_nbr,
                                      kernel_size=3,
                                      padding='same',
                                      strides=(2, 2),
                                      output_padding=(1, 1),
                                      data_format='channels_first')

    @tf.function
    def call(self, inputs, training=False):
        if type(inputs) is not dict or len(inputs) <= 1:
            raise Exception('inputs is not a tuple')
        x = inputs['x']
        blur = inputs['blur']

        size = tuple(x.get_shape())
        window_train = size[0]
        window_predict = (blur.get_shape())[0]

        hidden_state = None
        output_inner = []
        for t in range(window_train):
            z = x[t, :]
            z = self.padding(z)
            z = tf.nn.relu(self.conv1(z))
            z = self.padding(z)
            z = tf.nn.relu(self.conv2(z))
            z = self.padding(z)
            z = tf.nn.relu(self.conv3(z))
            z = tf.nn.relu(self.conv4(z))

            hidden_state = self.conv_recurrent_cell(z)

            y = hidden_state[0]

            y = tf.nn.relu(self.convd4(y))
            y = tf.nn.relu(self.convd3(y))
            y = tf.nn.relu(self.convd2(y))
            y = self.convd1(y)

        output_inner.append(y)

        for t in range(window_predict - 1):  # loop for every step

            if self.diff:
                z = y + blur[t, :]
            else:
                z = y

            # coder
            z = self.padding(z)
            z = tf.nn.relu(self.conv1(z))
            z = self.padding(z)
            z = tf.nn.relu(self.conv2(z))
            z = self.padding(z)
            z = tf.nn.relu(self.conv3(z))
            z = tf.nn.relu(self.conv4(z))

            # recurrent
            hidden_state = self.conv_recurrent_cell(z, hidden_state)

            y = hidden_state[0]

            y = tf.nn.relu(self.convd4(y))
            y = tf.nn.relu(self.convd3(y))
            y = tf.nn.relu(self.convd2(y))
            y = self.convd1(y)

            output_inner.append(y)

        return tf.stack(output_inner)
