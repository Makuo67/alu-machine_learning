#!/usr/bin/env python3
""" Performs convolution on grey scale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function to convolve the image"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    grayscale_images = 0.2989 * images[:, :, :, 0] + 0.5870 * images[:, :, :, 1] + 0.1140 * images[:, :, :, 2]

    # Calculate the output image dimensions
    oh = h - kh + 1
    ow = w - kw + 1

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(grayscale_images[:, i: i + kh, j: j + kw] * kernel, axis=(1, 2))
    
    return output
