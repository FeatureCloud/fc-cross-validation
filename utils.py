import numpy as np
import pandas as pd
from FeatureCloud.app.engine.app import LogLevel, app
from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server
from bottle import Bottle


def run(host='localhost', port=5000):
    """ run the docker container on specific host and port.

    Parameters
    ----------
    host: str
    port: int

    """

    app.register()
    server = Bottle()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host=host, port=port)


def save_numpy(file_name, features, labels, target):
    format = file_name.strip().split(".")[1].lower()
    save = {"npy": np.save, "npz": np.savez_compressed}
    if target == "same-sep":
        save[format](file_name, np.array([features, labels]))
    elif target == "same-last":
        samples = [np.append(features[i], labels[i]) for i in range(features.shape[0])]
        save[format](file_name, samples)
    elif target.strip().split(".")[1].lower() == 'npy':
        np.save(file_name, features)
        np.save(target, labels)
    elif target.strip().split(".")[1].lower() in 'npz':
        np.savez_compressed(file_name, features)
        np.savez_compressed(target, labels)
    else:
        return None


def load_numpy(file_name):
    ds = np.load(file_name, allow_pickle=True)
    format = file_name.strip().split(".")[1].lower()
    if format == "npz":
        return ds['arr_0']
    return ds


def sep_feat_from_label(ds, target):
    if target == 'same-sep':
        return pd.DataFrame({"features": [s for s in ds[0]], "label": ds[1]})
    elif target == 'same-last':
        return pd.DataFrame({"features": [s[:-1] for s in ds], "label": [s[-1] for s in ds]})
    elif target.strip().split(".")[1].lower() in ['npy', 'npz']:
        labels = load_numpy(target)
        return pd.DataFrame({"features": [s for s in ds], "label": labels})
    else:
        return None