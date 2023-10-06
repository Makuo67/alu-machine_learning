#!/usr/bin/env python3
""" Performs convolution on grey scale images"""
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """Function to convolve the image"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    #Calculate the output image dimensions
    oh = h - kh + 1
    ow = w - kw + 1

    output = np.zeros((m, oh, ow))

    for i in range(m):
        for y in range(oh):
            for x in range(ow):
                patch = images[i, y: y + kh, x: x + kw]
                output[i, y, x] = np.sum(patch * kernel)

    return output
