from neural_net import model
from data_processing import process
from tkinter import *
from skimage.transform import resize
from matplotlib import pyplot as plt

import numpy as np

CANVAS_WIDTH = 650
CANVAS_HEIGHT = 400

MOUSE_SIZE = 10


def is_in_range(x, y):
    return CANVAS_WIDTH > x >= 0 and CANVAS_HEIGHT > y >= 0


def predict_image(image_data):
    image_resized = np.flip(np.rot90(resize(image_data, (28, 28))), axis=0)
    plot_mnist_image(image_resized)


def plot_mnist_image(img):
    pixels = np.array(img, dtype='float32')
    pixels = pixels.reshape((28, 28))

    plt.imshow(pixels, cmap='Blues')
    plt.show()


def init_canvas():
    def process_image(event):
        predict_image(image_data)
        image_data.fill(0)
        canvas.delete("all")

    def mouse_move(event):
        create_gradient(event.x, event.y)

    def create_gradient(x, y):
        for i in range(1, MOUSE_SIZE):
            if is_in_range(x+i, y+i) and is_in_range(x-i, y-i):
                canvas.create_oval(x-i, y-i, x+i, y+i, fill='#000000')
                update_grid(x, y, i)

        image_data[x][y] = 255

    def update_grid(x, y, offset):
        for i in range(x-offset, x+offset):
            image_data[i][y-offset] = 255 / offset
            image_data[i][y+offset] = 255 / offset

        for j in range(y-offset, y+offset):
            image_data[x-offset][j] = 255 / offset
            image_data[x+offset][j] = 255 / offset

    image_data = np.zeros((CANVAS_WIDTH, CANVAS_HEIGHT))

    window = Tk()
    window.title("Image Net")
    canvas = Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, cursor="pencil")
    canvas.pack(expand=YES, fill=BOTH)

    window.bind('<B1-Motion>', mouse_move)
    window.bind('<Return>', process_image)

    window.mainloop()

def main():
    cached_weights = process.read_weights_from_file("model_weights/weights.txt")
    cached_bias = process.read_weights_from_file("model_weights/bias.txt")
    network = model.SingleLayerModel(cached_weights, cached_bias)

    init_canvas()


if __name__ == "__main__":
    main()

