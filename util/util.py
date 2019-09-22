import io

import numpy as np
from matplotlib import pyplot as plt
from util.constants import *
from PIL import Image, ImageChops
from resizeimage import resizeimage


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


def get_center_of_mass(data):
    x_sum = 0
    y_sum = 0

    for i in range(len(data)):
        for j in range(len(data[i])):
            x_sum += (i * data[i][j])
            y_sum += (j * data[i][j])
    return int(x_sum // np.sum(data)), int(y_sum // np.sum(data))


def process_image(image: Image):
    newImage = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 255)
    image = trim_whitespace(image.convert('L'))
    cover = resizeimage.resize_contain(image, [EDIT_SIZE, EDIT_SIZE])

    X = np.array([rgb[0] for rgb in list(cover.getdata())])
    X = invert_image(X, EDIT_SIZE)
    center_of_mass = get_center_of_mass(X)

    image_offset = np.subtract(14, center_of_mass).tolist()

    newImage.paste(cover, image_offset)
    return invert_image(np.array(list(newImage.getdata())), IMAGE_SIZE)

