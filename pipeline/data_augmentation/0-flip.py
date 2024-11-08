#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def flip_image(image):
    """
    Flips an image horizontally.

    Param:
        image (tf.Tensor): A 3D Tensor representing an image.

    Returns:
        tf.Tensor: The flipped image.
    """
    return tf.image.flip_left_right(image)


if __name__ == "__main__":
    doggies = tfds.load("stanford_dogs", split="train", as_supervised=True)
    
    # Shuffle and take 1 image to display
    for image, _ in doggies.shuffle(10).take(1):
        flipped_image = flip_image(image)
        plt.imshow(flipped_image)
        plt.show()

