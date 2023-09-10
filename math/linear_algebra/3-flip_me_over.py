#!/usr/bin/env python3

"""FUnction that returns the transpose of a matrix"""


def matrix_transpose(matrix):
    """Returns the transpose"""
    transpose = []
    for i in range(len(matrix[0])):
        transpose.append([])
    for row in matrix:
        for i in range(len(row)):
            transpose[i].append(row[i])
    return transpose
