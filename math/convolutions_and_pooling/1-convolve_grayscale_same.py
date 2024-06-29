#!/usr/bin/env python3
"""
this function performs same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs same convolution on grayscale images.

    Args:
        images (numpy.ndarray): Input grayscale images with shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant')

    output_h = h
    output_w = w
    convolved_images = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i + kh, j:j + kw] * kernel, axis=(1, 2))

    return convolved_images
