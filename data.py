"""
This module is used to load different datasets or synthesize data.
"""

import os

import numpy as np


def get_data(config):
    """
    Retrieve a dataset.

    Args:
        config: Configuration dictionary.

    Returns:
        x, y, s
    """

    dataset = config["type"]
    data_path = os.path.join(os.path.abspath(config["path"]), dataset)
    protected = config["protected"]

    if dataset == "compas":
        x, y, s = format_compas_data(data_path)
    else:
        raise RuntimeError(f"Unknown dataset {dataset}.")

    if config["protected_as_feature"]:
        x = np.concatenate((x, s.reshape(-1, 1)), axis=1)
    x = whiten(x)
    return x, y, s.astype(int)


def format_compas_data(data_path):
    """
    Load and preprocess the compas dataset.

    Args:
        data_path: The path to the data set without extension.

    Returns:
        x, y, s
    """
    raw_data = np.load(data_path + ".npz")
    x = raw_data["X"]
    y = raw_data["y"]
    s = raw_data["Z"][:, 0]
    y = to_zero_one(y)
    s = to_zero_one(s)
    return x, y, s


# -------------------------------------------------------------------------
# region HELPERS
# -------------------------------------------------------------------------
def whiten(data, columns=None, conditioning=1e-8):
    """
    Whiten various datasets in data dictionary.

    Args:
        data: Data array.
        columns: The columns to whiten. If `None`, whiten all.
        conditioning: Added to the denominator to avoid divison by zero.
    """
    if columns is None:
        columns = np.arange(data.shape[1])
    mu = np.mean(data[:, columns], 0)
    std = np.std(data[:, columns], 0)
    data[:, columns] = (data[:, columns] - mu) / (std + conditioning)
    return data


def to_zero_one(data):
    """Transform all binary columns with +/- 1 values to 0/1."""
    is_1d = False
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        is_1d = True
    for j in range(data.shape[1]):
        vals = np.unique(data[:, j])
        if len(vals) == 2:
            vals = sorted(vals)
            if vals[0] == -1 and vals[1] == 1:
                data[:, j] = (data[:, j] + 1 / 2).astype(int)
    return data.squeeze() if is_1d else data


# endregion
