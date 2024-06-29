#!/usr/bin/env python3
"""
A function that performs a convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):

    """
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_h = h + 2 * ph
    padded_w = w + 2 * pw

    # Calculate the output dimensions
    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1

    # Initialize the output array
    convolved_images = np.zeros((m, out_h, out_w))

    # Pad the images
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Perform the convolution
    for i in range(out_h):
        for j in range(out_w):
            slice_h = i * sh
            slice_w = j * sw
            image_slice = padded_images[
                :,
                slice_h:slice_h + kh,
                slice_w:slice_w + kw
            ]
            convolved_images[:, i, j] = np.sum(
                image_slice * kernel,
                axis=(1, 2)
            )

    return convolved_images
