#!/usr/bin/env python3
"""Function that slices a matrix along specific axes"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis."""

    # Base case: if one of the matrices is not a list, return None
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    # If the depth of the two matrices are different, return None
    if isinstance(mat1[0], list) != isinstance(mat2[0], list):
        return None

    # When axis is 0, concatenate the two matrices directly
    if axis == 0:
        return mat1 + mat2

    # For axis > 0, check if the current dimensions are the same
    if len(mat1) != len(mat2):
        return None

    # Recursively concatenate the next dimension down
    concatenated = []
    for a, b in zip(mat1, mat2):
        concatenated_part = cat_matrices(a, b, axis - 1)
        if concatenated_part is None:
            return None
        concatenated.append(concatenated_part)

    return concatenated
