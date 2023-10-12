#!/usr/bin/env python3

"""Function that concatenates two matrices along a specific axis"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis."""

    # Base case for 1D list (or the deepest dimension in multi-D lists)
    if type(mat1[0]) not in [list, tuple] and axis == 0:
        return mat1 + mat2

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [cat_matrices(
            mat1[i], mat2[i], axis=0) for i in range(len(mat1))]

    for i in range(axis):
        if len(mat1) != len(mat2):
            return None
        mat1 = mat1[0]
        mat2 = mat2[0]

    # Recursive case for multi-dimensional lists
    if len(mat1) != len(mat2):
        return None
    return [cat_matrices(
        mat1[i], mat2[i], axis=axis-1) for i in range(len(mat1))]
