#!/usr/bin/env python3
"""Function that concatenates 2 matrix at an axis"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenation"""
    return np.concatenate((mat1, mat2), axis=axis)
