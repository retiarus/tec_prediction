import os
import pickle

import numpy as np
import pandas as pd
import torch.utils.data as data
import h5py
import redis

class SequenceLoader(data.Dataset):
    """Main Class for Image Folder loader."""
    def __init__(self,
                 name,
                 path_files,
                 seq_length_min,
                 step_min,
                 window_train,
                 window_predict,
                 data=None):
        """Init function."""

        self.name = name
        self.path_files = path_files
        self.seq_length_min = seq_length_min
        self.step_min = step_min
        self.window_train = window_train
        self.window_predict = window_predict
        self.shape = (72, 72)
        self.dtype = np.float32
        self.data = data
        self.step = int(step_min / 10)
        self.cache = True

        if self.data not in ['tec', 'scin', 'tec+scin']:
            raise Exception(f'No valid data')

        # try load samples from file, if not generate samples from database
        file = f'samples_{self.name}_{self.seq_length_min}_{self.step_min}_{self.data}.pkl'
        path = os.path.join('./data', file)
        if os.path.isfile(path):
            self.samples = pickle.load(open(path, "rb"))
            self.df = pd.read_pickle(
                os.path.join('./data', f'df_{name}_{self.data}.pkl'))
        else:
            raise Exception(f'No valid df and samples list find in {file}')

    def load(self, index):
        r = redis.Redis(host="localhost")
        try:
            r.ping()
        except redis.ConnectionError as e:
            self.cache = False

        with h5py.File(f'./data/data_{self.name}_{self.data}.h5', 'r' ) as hdf:
            while True:
                try:
                    index_start, index_end = self.samples[index]

                    df_aux = self.df[index_start:index_end]

                    sample = np.zeros((int(self.seq_length_min / self.step_min), 1,
                                       self.shape[0], self.shape[1]))
                    for idx, aux in enumerate(df_aux.itertuples()):
                        if idx % self.step == 0:
                            key = os.path.basename(aux.path).replace('txt', 'npy').replace('.npy', '')

                            if self.cache:
                                aux_array = self.load_from_cache(r, key)
                                if aux_array is None:
                                    aux_array = np.array(hdf.get(key))
                                    self.store_to_cache(r, key, aux_array)

                                sample[idx // self.step, 0, :, :] = aux_array
                            else:
                                sample[idx // self.step, 0, :, :] = np.array(hdf.get(key))

                    if np.any(np.isnan(sample)):
                        index = np.random.choice(np.arange(0, self.__len__()))
                    else:
                        break
                except KeyError as e:
                    index = np.random.choice(np.arange(0, self.__len__()))

        return sample.astype(np.float32), np.array([index])

    def store_to_cache(self, r, key, a):
        a_byte = a.tobytes()
        r.set(key, a_byte)

    def load_from_cache(self, r, key):
        a_byte = r.get(key)
        if a_byte is None:
            return
        else:
            a = np.frombuffer(a_byte)
            return a.astype(self.dtype).reshape(self.shape)

    def __getitem__(self, index):
        """Get item."""
        return self.load(index)

    def __len__(self):
        """Length."""
        return len(self.samples)
