#!/usr/bin/env python3
'''
   Function that rotates an image by 90 degrees counter-clockwise.
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise.

    Params:
        image (tf.Tensor): A 3D Tensor representing an image.

    Returns:
        tf.Tensor: The rotated image.
    """
    return tf.image.rot90(image, k=1)


if __name__ == '__main__':
    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    
    # Shuffle and take 1 image to display
    for image, _ in doggies.shuffle(10).take(1):
        rotated_image = rotate_image(image)
        plt.imshow(rotated_image)
        plt.show()

