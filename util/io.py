import numpy as np


def validate_inputs(argv):
    if len(argv) != 2 and len(argv) != 4:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python3 main.py <type> [<output file> <number of rows> "
              "<number of columns>]")
        exit(1)


def write_array_to_file(weights: np.ndarray, filename: str, fmt: str):
    np.savetxt(filename, weights, fmt=fmt)


def read_weights_from_file(weights_input_path: str):
    return np.loadtxt(weights_input_path)
