from tkinter import *
from util.constants import *
from skimage.transform import resize
from matplotlib import pyplot as plt

import numpy as np


def plot_mnist_image(img):
    pixels = np.array(img, dtype='float32')
    pixels = pixels.reshape((28, 28))

    plt.imshow(pixels, cmap='Blues')
    plt.show()


def predict_image(image_data):
    image_resized = np.flip(np.rot90(resize(image_data, (28, 28))), axis=0)
    plot_mnist_image(image_resized)
    return 1


def is_in_range(x, y):
    return CANVAS_WIDTH > x >= 0 and CANVAS_HEIGHT > y >= 0


def init_canvas():
    window = Tk()
    window.title("ImageNet")
    canvas = Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, cursor="pencil")
    canvas.pack(expand=YES, fill=BOTH)

    return window, canvas


def listen(window, canvas, network):
    def mouse_move(event):
        if event.x > 0 and event.y > 0:
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

    def process_image(event):
        predicted_val = predict_image(image_data)
        image_data.fill(0)
        canvas.delete("all")

    image_data = np.zeros((CANVAS_WIDTH, CANVAS_HEIGHT))

    window.bind('<B1-Motion>', mouse_move)
    window.bind('<Return>', process_image)
    window.mainloop()

