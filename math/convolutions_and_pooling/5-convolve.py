#!/usr/bin/env python3

"""
A function that performs a convolution on images using multiple kernels
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):

    """
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
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
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    output = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                slice_h = slice(i * sh, i * sh + kh)
                slice_w = slice(j * sw, j * sw + kw)
                image_slice = padded_images[:, slice_h, slice_w, :]
                kernel = kernels[:, :, :, k]
                output[:, i, j, k] = np.sum(image_slice * kernel,
                                            axis=(1, 2, 3))

    return output
