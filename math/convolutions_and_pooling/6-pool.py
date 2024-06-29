#!/usr/bin/env python3

"""
A function that performs pooling on images
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Returns: a numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    pooled = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            x_st = i * sh
            x_end = x_st + kh
            y_st = j * sw
            y_end = y_st + kw

            if mode == 'max':
                pooled[:, i, j, :] = np.max(images[:, x_st:x_end,
                                                   y_st:y_end, :],
                                            axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(images[:, x_st:x_end,
                                                    y_st:y_end, :],
                                             axis=(1, 2))

    return pooled
