import os

from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd

def read_data(path, sep):
    return pd.read_csv(path, sep=sep)


def create_splits(data, label, n_splits, stratify, shuffle, random_state, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if stratify and (label is not None):
        print(f'Use stratified kfold cv for label column {label}', flush=True)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        y = data.loc[:, label]
        data = data.drop(label, axis=1)
        for k, (train_indices, test_indices) in enumerate(cv.split(data, y)):
            path = f'{output_dir}/split_{str(k+1)}'
            os.mkdir(path)
            pd.DataFrame(data.iloc[train_indices], columns=data.columns).to_csv(f'{path}/train.csv', index=False)
            pd.DataFrame(data.iloc[test_indices], columns=data.columns).to_csv(f'{path}/test.csv', index=False)
    else:
        print(f'Use kfold cv', flush=True)
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for k, (train_indices, test_indices) in enumerate(cv.split(data)):
            path = f'{output_dir}/split_{str(k+1)}'
            os.makedirs(path, exist_ok=True)
            pd.DataFrame(data.iloc[train_indices], columns=data.columns).to_csv(f'{path}/train.csv', index=False)
            pd.DataFrame(data.iloc[test_indices], columns=data.columns).to_csv(f'{path}/test.csv', index=False)
