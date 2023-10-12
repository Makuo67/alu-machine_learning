#!/usr/bin/env python3
"""Function that calculates the determinant of a matrix"""


def determinant(matrix):
    """Check if matrix is a list of lists"""
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base case: 0x0 matrix
    if n == 0:
        return 1

    # Base case: 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursively calculate determinant for larger matrices
    det = 0
    for i in range(n):
        # Create submatrix excluding the current row and column
        submatrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        # Use the Laplace expansion to calculate determinant
        det += matrix[0][i] * determinant(submatrix) * (-1 if i % 2 else 1)

    return det
