#!/usr/bin/env python3

"""Calculates the shape of a matrix"""


def matrix_shape(matrix):
    if not matrix:
        return []

    shape = []

    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]

    return shape