#!/usr/bin/env python3
"""One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.
    Y: numpy.ndarray - numeric class labels
    classes: int - maximum number of classes
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    one_hot = np.eye(classes)[Y].T
    return one_hot
