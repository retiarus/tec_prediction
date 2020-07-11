import os
import pdb

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

MAX = 0
MIN = 0

for fase in ['train', 'test', 'validation']:
    df = pd.read_pickle(f'./df_{fase}_tec+scin.pkl')
    hdf = h5py.File(f'./data_{fase}_tec+scin.h5')

    df.columns = ["tec", "scin"]

    keys = []
    for i in tqdm(df.itertuples()):
        aux = np.zeros((2, 72, 72))
        tec = i.tec
        scin = i.scin
        tec = os.path.basename(tec).replace('.txt', '').replace('.npy', '')
        scin = os.path.basename(scin).replace('.txt', '').replace('.npy', '')

        aux[0, :, :] = np.load(os.path.join('/mnt/data/resized', tec + '.npy'))
        aux[1, :, :] = np.load(os.path.join('/mnt/data/resized_type_1', scin + '.npy'))

        key = scin.replace("sci_", "")
        keys.append(key)
        hdf.create_dataset(key, data=aux.astype(np.float32))

        aux_max = np.max(aux[0, :, :])
        aux_min = np.min(aux[0, :, :])

        if aux_max > MAX:
            MAX = aux_max

        if aux_min < MIN:
            MIN = aux_min

    hdf.close()

    df['path'] = keys
    df.drop(['tec', 'scin'], inplace=True, axis=1)
    df.to_pickle(f'./df_{fase}_tec+scin.pkl')

    print(f'max => {MAX}')
    print(f'min => {MIN}')
