#!/usr/bin/env python3


"""
A function  that performs a convolution on images with channels
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):

    """
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    padded_h = h + 2 * ph
    padded_w = w + 2 * pw

    out_h = (padded_h - kh) // sh + 1
    out_w = (padded_w - kw) // sw + 1

    padded_images = np.pad(images,
                           ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw
            output[:, i, j] = np.sum(
                padded_images[:, vert_start:vert_end,
                              horiz_start:horiz_end, :] * kernel,
                axis=(1, 2, 3)
            )

    return output
