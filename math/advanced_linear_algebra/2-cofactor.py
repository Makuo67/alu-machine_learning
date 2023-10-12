#!/usr/bin/env python3
"""Function that calculates the co-factor of a matrix"""


def determinant(mat):
    """Det"""
    if len(mat) == 0:
        return 1
    if len(mat) == 1:
        return mat[0][0]
    if len(mat) == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    det = 0
    for i in range(len(mat)):
        submatrix = [row[:i] + row[i+1:] for row in mat[1:]]
        det += mat[0][i] * determinant(submatrix) * (-1 if i % 2 else 1)
    return det


def minor(matrix, i, j):
    """Returns the minor of the matrix
    after removing the ith row and jth column."""
    return [row[:j] + row[j+1:] for k, row in enumerate(matrix) if k != i]


def cofactor(matrix):
    """Returns the cofactor matrix of the given matrix."""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a list of lists")
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    return [[((-1) ** (i + j)) * determinant(
        minor(matrix, i, j)) for j in range(n)] for i in range(n)]
