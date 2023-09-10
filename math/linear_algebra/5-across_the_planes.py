#!/usr/bin/env python3

"""FUnction that adds two matrices"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices"""
    new_array = []
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        if isinstance(mat1[i], list) and len(mat1[i]) != len(mat2[i]):
            return None
        elif not isinstance(mat1[i], list):
            new_array.append(mat1[i] + mat2[i])
        elif isinstance(mat1[i], list):
            new_array.append([])
            for j in range(len(mat1[i])):

                new_array[i].append(mat1[i][j] + mat2[i][j])
    return new_array
