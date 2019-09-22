from tkinter import *
from util.constants import *
from skimage.transform import resize
from matplotlib import pyplot as plt
import util.util as util
import numpy as np


def predict_image(image_data, network):
    image_resized = np.flip(np.rot90(resize(image_data, (IMAGE_SIZE, IMAGE_SIZE))), axis=0)
    util.plot_mnist_image(image_resized)

    probabilities = network.call(image_resized.reshape((1, IMAGE_SIZE * IMAGE_SIZE)))
    return np.argmax(probabilities)


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
        predicted_val = predict_image(image_data, network)
        image_data.fill(0)
        canvas.delete("all")
        print(predicted_val)

    image_data = np.zeros((CANVAS_WIDTH, CANVAS_HEIGHT))

    window.bind('<B1-Motion>', mouse_move)
    window.bind('<Return>', process_image)
    window.mainloop()
