import glob
import tensorflow as tf

from canvas import *
from neural_net.model import ImageNet, train
from util import image_tools as imt
from util import io as io
from util.constants import *


def predict_images(images, network, filename):
    def run_network(X):
        probabilities = network.call(X.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1))
        return np.argmax(probabilities)

    images = np.asarray(images, dtype='float64')
    labels = np.apply_along_axis(run_network, 1, images)
    io.write_array_to_file(labels, filename, "%d")


def read_images(directory):
    image_list = []
    for filename in glob.glob(directory + '*', recursive=True):
        try:
            im = imt.process_image(Image.open(filename))
            im = im.reshape(-1)
            image_list.append(im)
        except OSError:
            pass
    return image_list


def main(argv):
    """
    Main function to invoke ImageNet.
    :param argv: type [<input directory>, <output_file_name>]
    """
    io.validate_inputs(argv)
    tf.keras.backend.set_floatx('float64')
    model = ImageNet()

    model.load_weights('./neural_net/checkpoints/ImageNet')

    # 1 for reading images from directory 0 for starting GUI to draw images
    if int(argv[1]) == 1:
        directory, filepath = argv[2], argv[3]
        image_list = read_images(directory)

        if len(image_list) > 0:
            predict_images(image_list, model, filepath)
        else:
            print("ERROR: No valid images in given directory")
            exit(1)
    elif int(argv[1]) == 0:
        window, canvas = init_canvas()
        listen(window, canvas, model)
    else:
        print("ERROR: Invalid type")
        exit(1)


if __name__ == "__main__":
    main(sys.argv)

