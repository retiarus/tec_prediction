import os

import numpy as np

import h5py
import pandas as pd
from tqdm import tqdm

for fase in ['train', 'test', 'validation']:
    df = pd.read_pickle(f'./df_{fase}_tec.pkl')
    hdf = h5py.File(f'./data_{fase}_tec.h5' )

    for i in tqdm(df['path']):
        name = os.path.basename(i).replace('.txt', '').replace('.npy', '')
        aux = np.load(os.path.join('/mnt/data/resized', name + '.npy')).astype(np.float32)
        hdf.create_dataset(name, data=aux)

    hdf.close()
