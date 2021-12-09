"""
    FeatureCloud Cross Validation Application
    Copyright 2021 Julian Spath and Mohammad Bakhtiari. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import ConfigState
from FeatureCloud.engine.app import app_state, Role, AppState, LogLevel
from FeatureCloud.engine.app import State as op_state
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import os
from utils import save_numpy, load_numpy, sep_feat_from_label

name = 'cross_validation'


@app_state(name='initial', role=Role.BOTH, app_name=name)
class LoadAndSplit(ConfigState.State):

    def register(self):
        self.register_transition('WriteResults', Role.BOTH)

    def run(self):
        self.lazy_init()
        self.read_config()
        self.finalize_config()
        self.store('format', self.config['local_dataset']['data'].lower().split(".")[-1].strip())
        self.store('sep', self.config['local_dataset']['sep'].strip())
        self.store('target', self.config['local_dataset']['target_value'].strip())
        df = self.read_data()
        # Creat placeholders for output files
        output_folder = f"{self.output_dir}/{self.config['split_dir']}"
        train, test = [], []
        for i in range(self.config['n_splits']):
            os.makedirs(f"{output_folder}/{i}")
            train.append(f"{output_folder}/{i}/{self.config['result']['train']}")
            test.append(f"{output_folder}/{i}/{self.config['result']['test']}")
        self.store('output_files', {'train': train, 'test': test})
        self.update(progress=0.1)
        self.store('splits', self.create_splits(df))
        self.update(progress=0.5)
        return 'WriteResults'

    def read_data(self):
        file_name = self.load('input_files')['data'][0]
        format = self.load('format')
        if format in ["npy", "npz"]:
            df = self.load_numpy_files(file_name)
        elif format in ["csv", "txt"]:
            df = pd.read_csv(file_name, sep=self.config['local_dataset']['sep'])
        else:
            self.app.log(f"{format} file types are not supported", LogLevel.ERROR)
            self.update(state=op_state.ERROR)
        return df

    def load_numpy_files(self, file_name):
        ds = load_numpy(file_name)
        target_value = self.config['local_dataset'].get('target_value', False)
        if target_value:
            df = sep_feat_from_label(ds, target_value)
            if df is None:
                self.app.log(f"{target_value} is not supported", LogLevel.ERROR)
                self.update(state=op_state.ERROR)
            return df
        else:
            self.log("For NumPy files, the format of target value should be mentioned through `target_value` "
                         "key in config file", LogLevel.ERROR)
            self.update(state=op_state.ERROR)

    def create_splits(self, data):
        splits = []
        if self.config['stratify'] and (self.config['local_dataset']['target_value'] is not None):
            self.log(f'Use stratified kfold cv for label column', LogLevel.DEBUG)
            cv = StratifiedKFold(n_splits=self.config['n_splits'], shuffle=self.config['shuffle'],
                                 random_state=self.config['random_state'])
            y = data.loc[:, self.config['local_dataset']['target_value']]
            data = data.drop(self.config['local_dataset']['target_value'], axis=1)
            for train_indices, test_indices in cv.split(data, y):
                train_df = pd.DataFrame(data.iloc[train_indices], columns=data.columns)
                test_df = pd.DataFrame(data.iloc[test_indices], columns=data.columns)
                splits.append([train_df, test_df])
        else:
            self.log(f'Use kfold cv', LogLevel.DEBUG)
            cv = KFold(n_splits=self.config['n_splits'], shuffle=self.config['shuffle'],
                       random_state=self.config['random_state'])
            for train_indices, test_indices in cv.split(data):
                train_df = pd.DataFrame(data.iloc[train_indices], columns=data.columns)
                test_df = pd.DataFrame(data.iloc[test_indices], columns=data.columns)
                splits.append([train_df, test_df])
        return splits


@app_state(name='WriteResults', role=Role.BOTH)
class WriteResults(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self) -> str or None:
        files = zip(self.load('splits'),
                    self.load('output_files')['train'],
                    self.load('output_files')['test'])
        print(files)
        csv_writer = lambda filename, df: df.to_csv(filename, sep=self.load('sep'), index=False)
        np_lambda = lambda filename, df: save_numpy(filename,
                                                    df.iloc[:, 0].to_numpy(),
                                                    df.iloc[:, 1].to_numpy(),
                                                    self.load('target'))
        save = {"npy": np_lambda, "npz": np_lambda, "csv": csv_writer, "txt": csv_writer}
        progress = 0.5
        step = 0.4 / len(self.load('splits'))
        for [train_split, test_split], train_filename, test_filename in files:
            print(train_filename, test_filename)
            save[self.load('format')](train_filename, train_split)
            save[self.load('format')](test_filename, test_split)
            progress += step
            self.update(progress=progress)
        self.update(progress=1.0)
        return 'terminal'
