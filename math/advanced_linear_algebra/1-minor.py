#!/usr/bin/env python3
"""Function that calculates the minor matrix of a matrix"""


def minor(matrix):
    """Helper function to compute determinant"""
    def determinant(mat):
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

    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square and non-empty
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a list of lists")
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Compute minor matrix
    minor_mat = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = [row[:j] + row[j+1:] for k, row in enumerate(
                matrix) if k != i]
            minor_row.append(determinant(submatrix))
        minor_mat.append(minor_row)

    return minor_mat


mat1 = [[5]]
mat2 = [[1, 2], [3, 4]]
mat3 = [[1, 1], [1, 1]]
mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
mat5 = []
mat6 = [[1, 2, 3], [4, 5, 6]]

print(minor(mat1))
print(minor(mat2))
print(minor(mat3))
print(minor(mat4))
try:
    minor(mat5)
except Exception as e:
    print(e)
try:
    minor(mat6)
except Exception as e:
    print(e)
