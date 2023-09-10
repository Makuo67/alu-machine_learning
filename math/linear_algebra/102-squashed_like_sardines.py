#!/usr/bin/env python3

"""Function that concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Function"""
    import numpy as np
    try:
        # Check if mat1 and mat2 have the same shape along the specified axis
        if mat1.shape[:axis] != mat2.shape[:axis] or mat1.shape[axis+1:] != mat2.shape[axis+1:]:
            return None

        # Use numpy's concatenate function to concatenate the matrices along the specified axis
        result = np.concatenate((mat1, mat2), axis=axis)

        return result
    except ValueError:
        # Handle ValueError if concatenation is not possible
        return None
