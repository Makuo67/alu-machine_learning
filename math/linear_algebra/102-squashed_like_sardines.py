#!/usr/bin/env python3

"""Function that concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Function"""
    import numpy as np

    if np.shape(mat1) != np.shape(mat2):
        return None

    # Use numpy's concatenate function to concatenate the matrices along the specified axis
    result = np.concatenate((mat1, mat2), axis=axis)

    return result
