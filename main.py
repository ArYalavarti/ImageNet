from neural_net import model
import data_processing as dp
from util.constants import *
from matplotlib import pyplot as plt
from util.image_util import plot_mnist_image
import sys
from PIL import Image
import glob

import numpy as np


def predict_images(images, network):
    def run_network(X):
        probabilities = network.call(X.reshape(1, 784))
        return np.argmax(probabilities)

    images = np.asarray(images)
    labels = np.apply_along_axis(run_network, 1, images)
    print(labels)


def main(argv):
    cached_weights = dp.read_weights_from_file("model_weights/weights.txt")
    cached_bias = dp.read_weights_from_file("model_weights/bias.txt")
    network = model.SingleLayerModel(cached_weights, cached_bias)

    image_list = []
    for filename in glob.glob(argv[1] + '*', recursive=True):
        try:
            im = dp.process_image(Image.open(filename))
            im = im.reshape(-1)
            image_list.append(im)
        except OSError:
            pass

    if len(image_list) > 0:
        plot_mnist_image(image_list[0])
        predict_images(image_list, network)
    else:
        print("ERROR: No valid images in given directory")




if __name__ == "__main__":
    main(sys.argv)

