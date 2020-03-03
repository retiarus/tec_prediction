import os
import pdb
import pickle

import numpy as np

import pandas as pd
from pre_processing import blur_array, get_periodic
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self,
                 name,
                 path_files,
                 seq_length_min,
                 step_min,
                 window_train,
                 window_predict,
                 data,
                 batch_size,
                 to_fit,
                 diff=False,
                 shuffle=True):
        self.name = name
        self.path_files = path_files
        self.seq_length_min = seq_length_min
        self.step_min = step_min
        self.window_train = window_train
        self.window_predict = window_predict
        self.data = data
        self.batch_size = batch_size
        self.to_fit = to_fit
        self.shuffle = shuffle
        self.diff = diff
        self.n = 0
        self.TEC_MAP_SHAPE = (72, 72)

        if self.data not in ['tec', 'scin', 'tec+scin']:
            raise Exception(f'No valid data')

        # load samples from file if exists, if not generate
        file_x = f'x_samples_{self.name}_{self.seq_length_min}_{self.step_min}_{self.data}.pkl'
        file_y = f'y_samples_{self.name}_{self.seq_length_min}_{self.step_min}_{self.data}.pkl'
        path_x = os.path.join('./data', file_x)
        path_y = os.path.join('./data', file_y)
        if os.path.isfile(path_x) and os.path.isfile(path_y):
            self.x_samples = pickle.load(open(path_x, "rb"))
            self.y_samples = pickle.load(open(path_y, "rb"))
            self.df = pd.read_pickle(
                os.path.join('./data', f'df_{name}_{self.data}.pkl'))
        else:
            raise Exception(
                f'No valid df and samples list find in {file_x} and {file_y}')

        self.indexes = np.arange(len(self.x_samples))

        self.on_epoch_end()

    def load_sample(self, index_start, index_end):
        df_aux = self.df[index_start:index_end]
        seq_lenght = df_aux.shape[0]

        sample = np.zeros(
            (seq_lenght, 1, self.TEC_MAP_SHAPE[0], self.TEC_MAP_SHAPE[1]))
        for idx, aux in enumerate(df_aux.itertuples()):
            sample[idx, 0, :, :] = np.load(
                os.path.join(self.path_files,
                             os.path.basename(aux.path).replace('txt', 'npy')))

        return sample[:self.window_train, :, :, :]

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0

        return data

    def __len__(self):
        # Return the number of batches of the dataset
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index_batch):
        # Generate indexes of the batch
        indexes = self.indexes[index_batch *
                               self.batch_size:(index_batch + 1) *
                               self.batch_size]

        # Find list of IDs
        x_samples_temp = [self.x_samples[k] for k in indexes]
        y_samples_temp = [self.y_samples[k] for k in indexes]

        dict_x = self._generate_x(x_samples_temp)

        if self.to_fit:
            y = self._generate_y(y_samples_temp)
            if self.diff:
                return dict_x, y - dict_x.blur
            else:
                return dict_x, y
        else:
            return dict_x

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.indexes))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_x(self, x_samples_temp):
        x = np.empty((self.batch_size, self.window_train, 1,
                      self.TEC_MAP_SHAPE[0], self.TEC_MAP_SHAPE[1]))

        for i, sample in enumerate(x_samples_temp):
            index_start, index_end = sample

            x[i, :, :, :, :] = self.load_sample(index_start, index_end)

        x = x.transpose(1, 0, 2, 3, 4)
        x_periodic = get_periodic(x, self.window_predict)
        x_blur = blur_array(x_periodic)

        return {
            'x': x.astype('np.float32'),
            'blur': x_blur.astype('np.float32')
        }

    def _generate_y(self, y_samples_temp):
        y = np.empty((self.batch_size, self.window_predict, 1,
                      self.TEC_MAP_SHAPE[0], self.TEC_MAP_SHAPE[1]))

        for i, sample in enumerate(y_samples_temp):
            index_start, index_end = sample

            y[i, :, :, :, :] = self.load_sample(index_start, index_end)

        return y.astype('np.float32')

    def len(self):
        return len(self.x_samples)
