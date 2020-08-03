import os
import pdb
import pickle
from os.path import basename, isfile, join

import h5py
import numpy as np
import pandas as pd
import torch.utils.data as data
from numpy import arange, array, zeros
from redis import ConnectionError, Redis
from torch import Tensor

from station import is_equinox, is_summer, is_winter


class SequenceLoader(data.Dataset):
    """Main Class for Image Folder loader."""
    def __init__(self,
                 name,
                 path_files,
                 seq_length_min,
                 step_min,
                 window_train,
                 window_predict,
                 data=None,
                 station=None):
        """Init function."""

        self.name = name
        self.path_files = path_files
        self.seq_length_min = seq_length_min
        self.step_min = step_min
        self.seq_length = int(seq_length_min / self.step_min)
        self.window_train = window_train
        self.window_predict = window_predict
        self.data = data
        self.step = int(step_min / 10)
        self.cache = True

        if self.data not in ['tec', 'scin', 'tec+scin']:
            raise Exception(f'No valid data')

        if self.data == 'tec+scin':
            self.input_channel = 2
            self.shape = (2, 72, 72)
            self.dtype = np.float32
        else:
            self.input_channel = 1
            self.shape = (72, 72)
            self.dtype = np.float64

        self.seq_shape = (self.seq_length, self.input_channel,
                          self.shape[-2], self.shape[-1])

        # try load samples from file, if not generate samples from database
        file = f'samples_{self.name}_{self.seq_length_min}_{self.step_min}_{self.data}.pkl'
        path = join('./data', file)
        if isfile(path):
            self.samples = pickle.load(open(path, "rb"))
            self.df = pd.read_pickle(
                join('./data', f'df_{name}_{self.data}.pkl'))
        else:
            raise Exception(f'No valid df and samples list find in {file}')

        if station == 'all':
            pass
        elif station == 'summer':
            self.samples = [i for i in self.samples if is_summer(i)]
        elif station == 'winter':
            self.samples = [i for i in self.samples if is_winter(i)]
        elif station == 'equinox':
            self.samples = [i for i in self.samples if is_equinox(i)]
        else:
            raise Exception(f'Invalid station => {station}')

        self.samples = np.array(self.samples)[np.random.choice(len(self.samples), 1000)]

    def load(self, index):
        r = Redis(host="localhost")
        try:
            r.ping()
        except ConnectionError as e:
            self.cache = False

        with h5py.File(f'./data/data_{self.name}_{self.data}.h5', 'r') as hdf:
            while True:
                try:
                    index_start, index_end = self.samples[index]

                    df_aux = self.df[index_start:index_end]

                    sample = zeros(self.seq_shape)

                    for idx, aux in enumerate(df_aux.itertuples()):
                        if idx % self.step == 0:
                            key = basename(aux.path).replace('txt', 'npy')
                            key = key.replace('.npy', '')

                            aux_array = None
                            if self.cache:
                                aux_array = self.load_from_cache(r, key)
                                if aux_array is None:
                                    aux_array = array(hdf.get(key))

                                    if self.data == "tec+scin" or self.data == "tec":
                                        aux_array[0, :, :][aux_array[0, :, :] == -np.inf] = 0
                                        aux_array[0, :, :][aux_array[0, :, :] < 0] = np.quantile(aux_array[0, :, :], 0.10)
                                        aux_array[0, :, :] /= 250

                                    self.store_to_cache(r, key, aux_array)
                            else:
                                aux_array = array(hdf.get(key))
                                if self.data == "tec+scin" or self.data == "tec":
                                    aux_array[0, :, :][aux_array[0, :, :] == -np.inf] = 0
                                    aux_array[0, :, :][aux_array[0, :, :] < 0] = np.quantile(aux_array[0, :, :], 0.10)
                                    aux_array[0, :, :] /= 250


                            # "tec+scin" data is store in (2, 72, 72) shape
                            # and the others in (72, 72), so for this it's
                            # necessary define the channel to atribute to
                            # sample
                            # TODO change from (72, 72) to (1, 72, 72)
                            if self.data == "tec+scin":
                                sample[idx // self.step, :, :, :] = aux_array
                            else:
                                sample[idx // self.step, 0, :, :] = aux_array

                    # test for empty array
                    if np.any(np.isnan(sample)):
                        index = np.random.choice(arange(0, self.__len__()))
                    else:
                        break

                except KeyError as e:
                    index = np.random.choice(np.arange(0, self.__len__()))

        X = Tensor(sample[0:self.window_train, :, :, :].astype(np.float32))
        y = Tensor(sample[self.window_train:, :, :, :].astype(np.float32))
        return X, y

    def store_to_cache(self, r, key, a):
        a_byte = a.reshape(-1).tobytes()
        r.set(key, a_byte)

    def load_from_cache(self, r, key):
        a_byte = r.get(key)
        if a_byte is None:
            return
        else:
            a = np.frombuffer(a_byte, dtype=self.dtype)
            try:
                aux = a.astype(self.dtype).reshape(self.shape)
                return aux
            except ValueError as e:
                return

    def __getitem__(self, index):
        """Get item."""
        return self.load(index)

    def __len__(self):
        """Length."""
        return len(self.samples)
