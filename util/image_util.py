import numpy as np
from matplotlib import pyplot as plt
from util.constants import *
from PIL import Image, ImageChops


def invert_image(img: Image, size):
    return np.abs(1 - (img.reshape((size, size)) / 255))


def plot_mnist_image(img):
    pixels = np.array(img, dtype='float32')
    pixels = pixels.reshape((IMAGE_SIZE, IMAGE_SIZE))

    plt.imshow(pixels, cmap='Greys_r')
    plt.show()


def trim_whitespace(image: Image):
    background = Image.new(image.mode, image.size, 255)
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)
