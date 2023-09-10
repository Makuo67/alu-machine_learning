#!/usr/bin/env python3

"""FUnction that adds two matrices"""


def add_matrices(mat1, mat2):
    """Check if mat1 and mat2 have the same shape"""
    if mat1.shape != mat2.shape:
        return None

    # Perform element-wise addition of the two matrices
    result = mat1 + mat2

    return result
