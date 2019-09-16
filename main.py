from neural_net import model
from data_processing import process
from tkinter import *
from skimage.transform import resize
from matplotlib import pyplot as plt

import numpy as np

CANVAS_WIDTH = 650
CANVAS_HEIGHT = 400

MOUSE_SIZE = 10

def init_canvas():
    image_data = np.zeros((CANVAS_WIDTH, CANVAS_HEIGHT))

    window = Tk()
    window.title("Image Net")
    canvas = Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, cursor="pencil")
    canvas.pack(expand=YES, fill=BOTH)

    window.mainloop()

def main():
    cached_weights = process.read_weights_from_file("model_weights/weights.txt")
    cached_bias = process.read_weights_from_file("model_weights/bias.txt")
    network = model.SingleLayerModel(cached_weights, cached_bias)

    init_canvas()


if __name__ == "__main__":
    main()

