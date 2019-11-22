import numpy as np
import gzip
import tensorflow as tf


def validate_inputs(argv):
    if len(argv) != 2 and len(argv) != 4:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python3 main.py <mode> [<image directory> <output file>]")
        exit(1)


def write_array_to_file(weights: np.ndarray, filename: str, fmt: str):
    np.savetxt(filename, weights, fmt=fmt)


def read_weights_from_file(weights_input_path: str):
    return np.loadtxt(weights_input_path)


def read_MNIST_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.reshape(x_train, [-1, 28, 28, 1])
    x_test = np.reshape(x_test, [-1, 28, 28, 1])

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return train_ds, test_ds
