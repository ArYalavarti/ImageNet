from neural_net import model
from data_processing import process
from canvas import init_canvas, listen


def main():
    cached_weights = process.read_weights_from_file("model_weights/weights.txt")
    cached_bias = process.read_weights_from_file("model_weights/bias.txt")
    network = model.SingleLayerModel(cached_weights, cached_bias)

    window, canvas = init_canvas()
    listen(window, canvas, network)


if __name__ == "__main__":
    main()

