#!/usr/bin/env python3
'''
    Function that performs a random crop of an image.
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Params:
        image (tf.Tensor): A 3D Tensor representing an image.
        size (tuple): The size of the crop (height, width, channels).

    Returns:
        tf.Tensor: The cropped image.
    """
    return tf.image.random_crop(image, size=size)


if __name__ == "__main__":
    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    
    # Shuffle and take 1 image to display
    for image, _ in doggies.shuffle(10).take(1):
        cropped_image = crop_image(image, (200, 200, 3))
        plt.imshow(cropped_image)
        plt.show()

