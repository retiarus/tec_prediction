import os
import pandas as pd
import numpy as np
import h5py

from tqdm import tqdm

for fase in ['train']:
    df = pd.read_pickle(f'./df_{fase}_scin.pkl')
    hdf = h5py.File(f'./data_{fase}_scin.h5' )

    for i in tqdm(df['path']):
        name = os.path.basename(i).replace('.npy', '')
        aux = np.load(os.path.join('/scratch/ampemi/pedro.santos2/resized_type_1', name + '.npy'))
        hdf.create_dataset(name, data=aux)

    hdf.close()
