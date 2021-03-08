import shutil
import threading
import time

import yaml

from app.algo import read_data, create_splits


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Will stop execution when True
        self.status_finished = False

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Directories, input files always in INPUT_DIR. Write your output always in OUTPUT_DIR
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        # === Variables from config.yml
        self.data_filename = None
        self.label_column = None
        self.sep = None
        self.train_output = None
        self.test_output = None
        self.split_dir = None
        self.n_splits = 5
        self.shuffle = True
        self.stratify = False
        self.random_state = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = "not started yet"
        self.dataset = None

    def handle_setup(self, client_id, master, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = master
        self.clients = clients
        print(f"Received setup: {self.id} {self.coordinator} {self.clients}", flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....", flush=True)
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...", flush=True)
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def read_config(self):
        print("Parsing config file...", flush=True)
        with open(self.INPUT_DIR + "/config.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)["fc_cross_validation"]
            print("config file", flush=True)

            self.data_filename = config["input"]["data"]
            print(self.data_filename)
            self.label_column = config["input"]["label_column"]
            print(self.label_column)
            self.sep = config["input"]["sep"]
            print(self.sep)

            self.train_output = config["output"]["train"]
            print(self.train_output)
            self.test_output = config["output"]["test"]
            print(self.test_output)
            self.split_dir = config["output"]["split_dir"]
            print(self.split_dir)

            self.n_splits = config["cross_validation"]["n_splits"]
            print(self.n_splits)
            self.shuffle = config["cross_validation"]["shuffle"]
            print(self.shuffle)
            self.stratify = config["cross_validation"]["stratify"]
            print(self.stratify)
            self.random_state = config["cross_validation"]["random_state"]
            print(self.random_state)

        print("Copy config file to outpur dir...", flush=True)
        shutil.copyfile(self.INPUT_DIR + "/config.yml", self.OUTPUT_DIR + "/config.yml")
        print(f'Read config file.', flush=True)

    def app_flow(self):
        # This method contains a state machine for the participant and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_create_splits = 3
        state_finish = 4

        # Initial state
        state = state_initializing
        while True:

            if state == state_initializing:
                self.progress = "initializing..."
                print("[CLIENT] Initializing...", flush=True)
                if self.id is not None:  # Test is setup has happened already
                    if self.coordinator:
                        print("I am the coordinator.", flush=True)
                    else:
                        print("I am a participating client.", flush=True)
                    state = state_read_input
                print("[CLIENT] Initializing finished.", flush=True)

            if state == state_read_input:
                self.progress = "read input..."
                print("[CLIENT] Read input...", flush=True)
                # Read the config file
                print("[CLIENT] Read config...", flush=True)
                self.read_config()
                print("[CLIENT] Read ta...", flush=True)
                self.dataset = read_data(f'{self.INPUT_DIR}/{self.data_filename}', sep=self.sep)
                # Here you could read in your input files
                state = state_create_splits
                print("[CLIENT] Read input finished.", flush=True)

            if state == state_create_splits:
                self.progress = "Create folds..."
                print("[CLIENT] Create folds...", flush=True)

                # Compute local results
                create_splits(self.dataset, self.label_column, self.n_splits, self.stratify, self.shuffle,
                              self.random_state, self.OUTPUT_DIR + "/" + self.split_dir)

                if self.coordinator:
                    self.data_incoming = ['DONE']
                    state = state_finish
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break
                print("[CLIENT] Create folds finished.", flush=True)

            if state == state_finish:
                print("Finishing", flush=True)
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    print("[CLIENT] All clients have finished. Finish coordinator.", flush=True)
                    self.status_finished = True
                    break

            time.sleep(1)


logic = AppLogic()
