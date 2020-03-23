import os
import pickle

import numpy as np
import pandas as pd
import torch.utils.data as data
import pdb


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
        self.data = data
        self.step = int(step_min / 10)

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
        while True:
            try:
                index_start, index_end = self.samples[index]

                df_aux = self.df[index_start:index_end]

                sample = np.zeros((int(self.seq_length_min / self.step_min), 1,
                                   self.shape[0], self.shape[1]))
                for idx, aux in enumerate(df_aux.itertuples()):
                    if idx // self.step == 0:
                        sample[idx, 0, :, :] = np.load(
                            os.path.join(
                                self.path_files,
                                os.path.basename(aux.path).replace('txt', 'npy')))

                if np.any(np.isnan(sample)):
                    index = np.random.choice(np.arange(0, self.__len__()))
                else:
                    break
            except KeyError as e:
                pdb.set_trace()
                index = np.random.choice(np.arange(0, self.__len__()))

        return sample.astype(np.float32), np.array([index])

    def __getitem__(self, index):
        """Get item."""
        return self.load(index)

    def __len__(self):
        """Length."""
        return len(self.samples)
