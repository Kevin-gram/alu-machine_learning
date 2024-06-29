#!/usr/bin/env python3

"""
A function that performs a convolution
on grayscale images with custom padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):

    """
    returns: numpy.ndarray containing the convolved images
    """
    # Extract dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Calculate new dimensions after padding
    new_h = h + 2 * ph
    new_w = w + 2 * pw

    # Pad images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Initialize output array
    output = np.zeros((m, new_h - kh + 1, new_w - kw + 1))

    # Perform convolution
    for i in range(new_h - kh + 1):
        for j in range(new_w - kw + 1):
            output[:, i, j] = (padded_images[:,
                                             i:i+kh,
                                             j:j+kw] * kernel).sum(axis=(1, 2))

    return output
