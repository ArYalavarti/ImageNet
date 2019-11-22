from __future__ import absolute_import, division, print_function, \
    unicode_literals

import tensorflow as tf
import numpy as np
from util import io as io

NUM_EPOCHS = 2


class ImageNet(tf.keras.Model):
    def __init__(self):
        super(ImageNet, self).__init__()
        # Define optimizer and learning rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # Define convolution layers
        self.Conv1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same',
                                            activation='relu')
        # Flattening and dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.Dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.Dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, images):
        conv_output = self.Conv1(images)
        flattened = self.flatten(conv_output)
        return self.Dense2(self.Dense1(flattened))

    def loss(self, labels, predictions):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=True))


def test(test_ds, network):
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for test_images, test_labels in test_ds:
        predictions = network(test_images)
        test_loss(network.loss(test_labels, predictions))
        test_accuracy(test_labels, predictions)

    print('Test Loss: {}, Test Accuracy: {}'
          .format(test_loss.result(), test_accuracy.result() * 100))


def train(train_ds, network):
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(NUM_EPOCHS):
        for images, labels in train_ds:
            with tf.GradientTape() as tape:
                predictions = network(images)
                l = network.loss(labels, predictions)
            trainable_variables = network.trainable_variables
            gradients = tape.gradient(l, trainable_variables)
            network.optimizer.apply_gradients(zip(gradients,
                                                  trainable_variables))
            train_loss(l)
            train_accuracy(labels, predictions)

        print('Epoch {}, Loss: {}, Accuracy: {}'
              .format(epoch + 1, train_loss.result(),
                      train_accuracy.result() * 100))

        train_loss.reset_states()
        train_accuracy.reset_states()


def initialize_image_model():
    train_ds, test_ds = io.read_MNIST_data()
    network = ImageNet()

    train(train_ds, network)
    test(test_ds, network)
    network.save_weights('./checkpoints/ImageNet')


if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')
    initialize_image_model()
