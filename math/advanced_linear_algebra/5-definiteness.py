#!/usr/bin/env python3
"""Function that calculates the definiteness of a matrix"""
import numpy as np


def definiteness(matrix):
    """Determine the definiteness of a matrix"""

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if the matrix is square
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric, as the methods below require symmetry
    if not np.allclose(matrix, matrix.T):
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    if np.all(eigenvalues < 0):
        return "Negative definite"
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
