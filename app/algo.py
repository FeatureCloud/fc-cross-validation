import os

from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import numpy as np


def read_data(path, sep, format):
    if format == "npy":
        x, y = np.load(path, allow_pickle=True)
        x, y = x.tolist(), y.tolist()
        df = pd.DataFrame(data=zip(x, y), index=None, columns=['data', 'label'])
    else:
        df = pd.read_csv(path, sep=sep)
    return df


def create_splits(data, label, n_splits, stratify, shuffle, random_state, output_dir, format="csv"):
    os.makedirs(output_dir, exist_ok=True)
    splits = {}
    if stratify and (label is not None):
        print(f'Use stratified kfold cv for label column {label}', flush=True)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        y = data.loc[:, label]
        data = data.drop(label, axis=1)
        for k, (train_indices, test_indices) in enumerate(cv.split(data, y)):
            path = f'{output_dir}/split_{str(k + 1)}'
            train_df = pd.DataFrame(data.iloc[train_indices], columns=data.columns)
            test_df = pd.DataFrame(data.iloc[test_indices], columns=data.columns)
            splits[path] = [train_df, test_df]
    else:
        print(f'Use kfold cv', flush=True)
        cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for k, (train_indices, test_indices) in enumerate(cv.split(data)):
            path = f'{output_dir}/split_{str(k + 1)}'
            train_df = pd.DataFrame(data.iloc[train_indices], columns=data.columns)
            test_df = pd.DataFrame(data.iloc[test_indices], columns=data.columns)
            splits[path] = [train_df, test_df]
    if format == "npy":
        for path, [train_df, test_df] in splits.items():
            os.mkdir(path)
            np.save(f"{path}/train.npy", [train_df.iloc[:, 0].to_numpy(), train_df.iloc[:, 1].to_numpy()])
            np.save(f"{path}/test.npy", [test_df.iloc[:, 0].to_numpy(), test_df.iloc[:, 1].to_numpy()])
    else:  # csv
        for path, [train_df, test_df] in splits.items():
            os.makedirs(path, exist_ok=True)
            train_df.to_csv(f'{path}/train.csv', index=False)
            test_df.to_csv(f'{path}/test.csv', index=False)
