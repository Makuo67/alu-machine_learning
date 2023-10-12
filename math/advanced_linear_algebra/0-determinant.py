#!/usr/bin/env python3
"""Function that calculates the determinant of a matrix"""


def determinant(matrix):
    """Determinant calculation"""
    # Handle the special case of 0x0 matrix
    if matrix == [[]]:
        return 1

    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

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


# mat0 = [[]]
# mat1 = [[5]]
# mat2 = [[1, 2], [3, 4]]
# mat3 = [[1, 1], [1, 1]]
# mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
# mat5 = []
# mat6 = [[1, 2, 3], [4, 5, 6]]

# print(determinant(mat0))
# print(determinant(mat1))
# print(determinant(mat2))
# print(determinant(mat3))
# print(determinant(mat4))
# try:
#     determinant(mat5)
# except Exception as e:
#     print(e)
# try:
#     determinant(mat6)
# except Exception as e:
#     print(e)
