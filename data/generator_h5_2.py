import os
import pandas as pd
import numpy as np
import h5py

from tqdm import tqdm

for fase in ['train']:
    df = pd.read_pickle(f'./df_{fase}_tec.pkl')
    hdf = h5py.File(f'./data_{fase}_tec.h5' )

    for i in tqdm(df['path']):
        name = os.path.basename(i).replace('.txt', '').replace('.npy', '')
        aux = np.load(os.path.join('/scratch/ampemi/pedro.santos2/resized', name + '.npy'))
        hdf.create_dataset(name, data=aux)

    hdf.close()
