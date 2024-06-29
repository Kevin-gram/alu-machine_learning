#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_same(images, kernel):
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for height and width
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # Pad the images with zeros on all sides
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Initialize the output convolved images array
    output = np.zeros((m, h, w))
    
    # Perform convolution using two for loops
    for i in range(h):
        for j in range(w):
            # Extract the region of interest
            region = padded_images[:, i:i+kh, j:j+kw]
            # Perform element-wise multiplication and summation
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
