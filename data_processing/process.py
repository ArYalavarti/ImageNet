import numpy as np


def write_weights_to_file(weights: np.ndarray, filename: str):
    np.savetxt(filename, weights)


def read_weights_from_file(weights_input_path: str):
    return np.loadtxt(weights_input_path)
