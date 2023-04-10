import shutil
import threading
import time

import yaml

from algo import read_data, create_splits
from FeatureCloud.app.engine.app import AppState, app_state, Role

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """

    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Initializing")
        if self.id is not None:  # Test if setup has happened already
            self.log(f"[CLIENT] Coordinator {self.is_coordinator}")
        self.log("[CLIENT] Initializing finished.")
        return 'read input'


@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """

    def register(self):
        self.register_transition('create splits', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Read input and config")
        self.read_config()
        self.log("[CLIENT] Read data...")
        dataset = read_data(f"{self.load('INPUT_DIR')}/{self.load('data_filename')}", sep=self.load('sep'))
        self.store('dataset', dataset)
        # Here you could read in your input files
        self.log("[CLIENT] Read input finished.")
        return 'create splits'

    def read_config(self):
        self.log("Parsing config file...")
        # === Directories, input files always in INPUT_DIR. Write your output always in OUTPUT_DIR
        self.store('INPUT_DIR', "/mnt/input")
        self.store('OUTPUT_DIR', "/mnt/output")

        # === Variables from config.yml
        self.store('n_splits', 5)
        self.store('shuffle', True)
        self.store('stratify', False)
        with open(self.load('INPUT_DIR') + "/config.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)["fc_cross_validation"]
            self.log("config file")

            self.store('data_filename', config["input"]["data"])
            self.log(self.load('data_filename'))
            self.store('label_column', config["input"]["label_column"])
            self.log(self.load('label_column'))
            self.store('sep', config["input"]["sep"])
            self.log(self.load('sep'))

            self.store('train_output', config["output"]["train"])
            self.log(self.load('train_output'))
            self.store('test_output', config["output"]["test"])
            self.log(self.load('test_output'))
            self.store('split_dir', config["output"]["split_dir"])
            self.log(self.load('split_dir'))

            self.store('n_splits', config["cross_validation"]["n_splits"])
            self.log(self.load('n_splits'))
            self.store('shuffle', config["cross_validation"]["shuffle"])
            self.log(self.load('shuffle'))
            self.store('stratify', config["cross_validation"]["stratify"])
            self.log(self.load('stratify'))
            self.store('random_state', config["cross_validation"]["random_state"])
            self.log(self.load('random_state'))

        self.log("Copy config file to outpur dir...")
        shutil.copyfile(self.load('INPUT_DIR') + "/config.yml", self.load('OUTPUT_DIR') + "/config.yml")
        self.log(f'Read config file.')


@app_state('create splits', Role.BOTH)
class CreateSplitsState(AppState):
    """
    Create folds.
    """

    def register(self):
        self.register_transition('terminal', Role.PARTICIPANT)
        self.register_transition('finish', Role.COORDINATOR)
         
    def run(self) -> str or None:
        self.log("[CLIENT] Create folds...")
        # Compute local results
        create_splits(self.load('dataset'), self.load('label_column'), self.load('n_splits'), self.load('stratify'), self.load('shuffle'),
                              self.load('random_state'), self.load('OUTPUT_DIR') + "/" + self.load('split_dir'))
        self.log("[CLIENT] Create folds finished.")
        self.send_data_to_coordinator('DONE')
        if self.is_coordinator:
            return 'finish'
        else:
            return 'terminal'


@app_state('finish', Role.COORDINATOR)
class FinishState(AppState):

    def register(self):
        self.register_transition('terminal', Role.COORDINATOR)
         
    def run(self) -> str or None:
        self.log("Finishing")
        self.gather_data()
        self.log("[CLIENT] All clients have finished. Finish coordinator.")
        return 'terminal'
