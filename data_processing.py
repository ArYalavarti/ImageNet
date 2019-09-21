from PIL import Image
from resizeimage import resizeimage
from util.image_util import *


def write_weights_to_file(weights: np.ndarray, filename: str):
    np.savetxt(filename, weights)


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

    newImage.paste(cover, np.subtract(14, center_of_mass).tolist())
    return invert_image(np.array(list(newImage.getdata())), IMAGE_SIZE)
