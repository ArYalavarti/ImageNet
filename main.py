from neural_net import model
import data_processing as dp
from util.constants import *
from matplotlib import pyplot as plt
from util.image_util import plot_mnist_image
import sys
from PIL import Image
import glob

import numpy as np


def validate_inputs(argv):
    if len(argv) != 2 and len(argv) != 4:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python3 main.py <type> [<output file> <number of rows> "
              "<number of columns>]")
        exit(1)


def predict_images(images, network):
    def run_network(X):
        probabilities = network.call(X.reshape(1, 784))
        return np.argmax(probabilities)

    images = np.asarray(images)
    labels = np.apply_along_axis(run_network, 1, images)
    print(labels)


def read_images(argv):
    image_list = []
    for filename in glob.glob(argv[2] + '*', recursive=True):
        try:
            im = dp.process_image(Image.open(filename))
            im = im.reshape(-1)
            image_list.append(im)
        except OSError:
            pass
    return image_list


def main(argv):
    """
    Main function to invoke ImageNet.
    :param argv: type [<input directory>, <output_file_name>]
    """
    validate_inputs(argv)

    cached_weights = dp.read_weights_from_file("model_weights/weights.txt")
    cached_bias = dp.read_weights_from_file("model_weights/bias.txt")
    network = model.SingleLayerModel(cached_weights, cached_bias)

    # Mode for reading images from directory
    if int(argv[1]) == 1:
        image_list = read_images(argv)

        if len(image_list) > 0:
            plot_mnist_image(image_list[0])
            predict_images(image_list, network)
        else:
            print("ERROR: No valid images in given directory")
            exit(1)
    # Mode for starting GUI to draw images
    elif int(argv[1]) == 0:
        pass
    else:
        print("ERROR: Invalid type")
        exit(1)


if __name__ == "__main__":
    main(sys.argv)

