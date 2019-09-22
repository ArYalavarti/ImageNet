import io
from tkinter import *

import numpy as np
from PIL import Image

import util.image_tools as imt
from util.constants import *


def predict_image(ps_image, network):
    im = imt.process_image(ps_image)
    imt.plot_mnist_image(im)

    probabilities = network.call(
        im.reshape((1, IMAGE_SIZE * IMAGE_SIZE)))

    return np.argmax(probabilities)


def is_in_range(x, y):
    return CANVAS_WIDTH > x >= 0 and CANVAS_HEIGHT > y >= 0


def init_canvas():
    window = Tk()
    window.title("ImageNet")
    canvas = Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT,
                    cursor="pencil")
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

    def export_postscript(event):
        ps = canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        canvas.delete("all")
        predict_image(img, network)
        print(3)

    window.bind('<B1-Motion>', mouse_move)
    window.bind('<Return>', export_postscript)
    window.mainloop()
