#!/usr/bin/env python3
"""
   Function that randomly shears an image.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def shear_image(image, intensity):
    """
    Randomly shears an image.

    Params:
        image (tf.Tensor): A 3D Tensor representing the image.
        intensity (float): The intensity of the shear to apply.

    Returns:
        tf.Tensor: The sheared image.
    """
    return tf.image.resize(image, (image.shape[0] + intensity, image.shape[1] + intensity))


if __name__ == "__main__":
    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        sheared_image = shear_image(image, intensity=50)
        plt.imshow(sheared_image)
        plt.show()

