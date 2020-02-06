import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg2
import torch.utils.data as data


def from_np_2_py_datetime(time):
    ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(
        1, 's')
    return datetime.utcfromtimestamp(ts)


class SequenceLoader(data.Dataset):
    """Main Class for Image Folder loader."""
    def __init__(self,
                 path_files,
                 seq_length_min,
                 step_min,
                 window_train,
                 window_predict,
                 state='training'):
        """Init function."""

        self.path_files = path_files
        self.seq_length_min = seq_length_min
        self.step_min = step_min
        self.window_train = window_train
        self.window_predict = window_predict

        # get filenames in database
        conn_parameters = {
            "database": "data-research",
            "user": "data-research",
            "password": "data-research",
            "host": "pve.falco.net",
            "port": 5432
        }

        # try to load dataframe with names from local file, if was not capable
        # load from postgres database
        if os.path.isfile(f'./df_filename.pkl'):
            self.df = pd.read_pickle(f'./df_filename.pkl')
        else:
            conn = psycopg2.connect(**conn_parameters)
            self.df = pd.read_sql_query(
                "select data_tecmap.file, data_tecmap.index_datetime from data_tecmap where data_tecmap.resized = True order by data_tecmap.index_datetime;",
                conn)
            conn.close()
            self.df.index = pd.to_datetime(self.df.index_datetime, utc=True)
            self.df.drop(['index_datetime'], inplace=True, axis=1)

            self.df.to_pickle(f'./df_filename.pkl')

        # try load samples from file, if not generate samples from database
        if os.path.isfile(
                f'./samples_{self.seq_length_min}_{self.step_min}.pkl'):
            samples = pickle.load(
                open(f'./samples_{seq_length_min}_{step_min}.pkl', "rb"))
        else:
            samples = self.generate_samples()
            pickle.dump(
                samples,
                open(f'./samples_{seq_length_min}_{step_min}.pkl', "wb"))

        # prepare samples for train, test, validation
        size = len(samples)
        point_1 = int(size * 0.5)
        point_2 = point_1 + int(size * 0.05)
        point_3 = point_2 + int(size * 0.2)
        point_4 = point_3 + int(size * 0.05)

        if state == 'training':
            self.samples = samples[:point_1]
        elif state == 'testing':
            self.samples = samples[point_2:point_3]
        elif state == 'validation':
            self.samples = samples[point_4:]

        self.TEC_MAP_SHAPE = (72, 72)
        self.state = state

    def load(self, index):
        index_start, index_end = self.samples[index]

        df_aux = self.df[index_start:index_end]

        sample = np.zeros((int(self.seq_length_min / self.step_min), 1,
                           self.TEC_MAP_SHAPE[0], self.TEC_MAP_SHAPE[1]))
        for idx, aux in enumerate(df_aux.itertuples()):
            sample[idx, 0, :, :] = np.load(
                os.path.join(self.path_files, aux.file.replace('txt', 'npy')))

        #return sample[:window_train, :, :, :], sample[window_train:, :, :, :]
        return sample, np.array([index])

    def generate_samples(self):
        start_index = self.df.index.values[0]
        start_index = from_np_2_py_datetime(start_index)
        end_index = self.df.index.values[-1]
        end_index = from_np_2_py_datetime(end_index)

        delta_seq_length = timedelta(minutes=self.seq_length_min -
                                     self.step_min)
        delta_step_min = timedelta(minutes=self.step_min)
        i = start_index
        samples = []
        while True:
            if i > end_index:
                break
            elif i + delta_seq_length > end_index + delta_step_min:
                break

            df_aux = self.df[i:i + delta_seq_length]
            if df_aux.shape[0] == int(self.seq_length_min / self.step_min):
                # sample ok
                samples.append((i, i + delta_seq_length))
            else:
                pass

            i += delta_step_min

        return samples

    def __getitem__(self, index):
        """Get item."""
        return self.load(index)

    def __len__(self):
        """Length."""
        return len(self.samples)
