from neural_net import model
import data_processing as dp
from util.constants import *
from matplotlib import pyplot as plt
from util.image_util import plot_mnist_image
import sys
from PIL import Image

import numpy as np


def main(argv):
    cached_weights = dp.read_weights_from_file("model_weights/weights.txt")
    cached_bias = dp.read_weights_from_file("model_weights/bias.txt")
    network = model.SingleLayerModel(cached_weights, cached_bias)

    im = dp.process_image(Image.open(argv[1]))
    plot_mnist_image(im)


if __name__ == "__main__":
    main(sys.argv)

