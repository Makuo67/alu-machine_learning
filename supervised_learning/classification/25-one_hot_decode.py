#!/usr/bin/env python3
"""One-Hot Decode"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.
    one_hot: numpy.ndarray - one-hot encoded matrix
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    return np.argmax(one_hot, axis=0)
