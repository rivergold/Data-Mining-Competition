import pandas as pd
import numpy as np
import pickle

for i in ['train', 'test']:
    file_name = '../dataset/{}.csv'.format(i)
    with open(file_name) as file:
        raw_data = pd.read_csv(file)
        x, y = None, None
        if i == 'train':
            x = raw_data.ix[:, 1:].values
            y = np.array(raw_data.ix[:, 0])
        else:
            x = raw_data.values
        input(x.shape)
        x = x.reshape(-1, 1, 28, 28)
        input(type(y))
        samples = {'data': x, 'label': y}
    file_name = '../dataset/{}.pkl'.format(i)
    with open(file_name, 'wb') as file:
        pickle.dump(samples, file)
