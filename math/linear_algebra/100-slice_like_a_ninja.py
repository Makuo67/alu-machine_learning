#!/usr/bin/env python3
"""Function that slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """Function"""
    import numpy as np
    slices = [slice(None)] * matrix.ndim

    # Update the slice objects for the specified axes
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)

    # Apply the slices to the matrix
    sliced_matrix = matrix[tuple(slices)]

    return sliced_matrix


# mat1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# print(np_slice(mat1, axes={1: (1, 3)}))
# print(mat1)
# mat2 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
#                  [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
#                  [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
# print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
# print(mat2)
