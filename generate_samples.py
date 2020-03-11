import argparse
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import psycopg2

import pandas as pd

# from colors import print_blue, print_green, print_red


def from_np_2_py_datetime(time):
    ts = (time - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(
        1, 's')
    return datetime.utcfromtimestamp(ts)


def break_samples_xy(samples, win_train, step_min):
    win_train_min = win_train * step_min
    delta_win_train = timedelta(minutes=win_train_min - step_min)
    delta_step_min = timedelta(minutes=step_min)

    x_samples = []
    y_samples = []
    for i in samples:
        index_start, index_end = i
        x_samples.append((index_start, index_start + delta_win_train))
        y_samples.append(
            (index_start + delta_win_train + delta_step_min, index_end))

    return x_samples, y_samples


def generate_samples(df, seq_length_min, step_min):
    start_index = df.index.values[0]
    start_index = from_np_2_py_datetime(start_index)
    end_index = df.index.values[-1]
    end_index = from_np_2_py_datetime(end_index)

    delta_seq_length = timedelta(minutes=seq_length_min - 10)
    delta_step_min = timedelta(minutes=10)
    i = start_index
    samples = []
    while True:
        if i > end_index:
            break

        if i + delta_seq_length > end_index + delta_step_min:
            break

        df_aux = df[i:i + delta_seq_length]
        if df_aux.shape[0] == int(seq_length_min / 10):
            # sample ok
            samples.append((i, i + delta_seq_length))
        else:
            pass

        i += delta_step_min

    return samples


def generate_data(data):
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
    conn = psycopg2.connect(**conn_parameters)
    if data == 'tec':
        query = f"select path, index_datetime from data_tecmap " \
                f"where data_tecmap.resized = True " \
                f"order by data_tecmap.index_datetime;"
    elif data == 'scin':
        query = f"select path, index_datetime from data_scintillationmap " \
                f"where data_scintillationmap.resized = True " \
                f"order by data_scintillationmap.index_datetime;"
    elif data == 'tec+scin':
        query = f"select data_file, index_datetime from data_tecmap " \
                f"where data_tecmap.resized = True " \
                f"order by data_tecmap.index_datetime;"
    else:
        print('No valid data choiced.')
        return

    df = pd.read_sql_query(query, conn)
    conn.close()

    df.index = pd.to_datetime(df.index_datetime, utc=True)
    df.drop(['index_datetime'], inplace=True, axis=1)

    # prepare samples for train, test, validation
    size = len(df)
    point_1 = int(size * 0.7)
    point_2 = point_1 + int(size * 0.03)
    point_3 = point_2 + int(size * 0.12)
    point_4 = point_3 + int(size * 0.03)

    df_train = df[:point_1]
    df_test = df[point_2:point_3]
    df_validation = df[point_4:]

    return df_train, df_test, df_validation


def generate_save_samples(seq_length_min, step_min, win_train, data):
    dfs = generate_data(data)
    names = ['train', 'test', 'validation']

    for name, df in zip(names, dfs):
        samples = generate_samples(df, seq_length_min, step_min)

        file = f'samples_{name}_{seq_length_min}_{step_min}_{data}.pkl'
        pickle.dump(samples, open(os.path.join('./data', file), "wb"))

        print(f"{name}, {data} has {len(samples)} samples")

        x_samples, y_samples = break_samples_xy(samples, win_train, step_min)
        file_x = f'x_samples_{name}_{seq_length_min}_{step_min}_{data}.pkl'
        file_y = f'y_samples_{name}_{seq_length_min}_{step_min}_{data}.pkl'
        pickle.dump(x_samples, open(os.path.join('./data', file_x), "wb"))
        pickle.dump(y_samples, open(os.path.join('./data', file_y), "wb"))

        df.to_pickle(os.path.join('./data/', f'df_{name}_{data}.pkl'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_length_min",
                        type=int,
                        default=7200,
                        help="train network or not")
    parser.add_argument("--step_min", type=int, default=10)
    parser.add_argument("--win_train", type=int, default=432)
    parser.add_argument("--source", type=str, help="source directory")
    parser.add_argument("--data", type=str, default='tec')
    args = parser.parse_args()

    generate_save_samples(args.seq_length_min, args.step_min, args.win_train,
                          args.data)


if __name__ == '__main__':
    main()
