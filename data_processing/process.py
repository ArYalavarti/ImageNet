import numpy as np


def write_weights_to_file(weights, filename):
    np.savetxt(filename, weights)


def read_weights_from_file(weights_input_path):
    return np.loadtxt(weights_input_path)