#!/usr/bin/env python3
"""Multivariate Normal distribution"""


import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """ Initialize MultiNormal """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        deviation = data - self.mean
        self.cov = np.dot(deviation, deviation.T) / (n - 1)

    def pdf(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Compute terms for PDF formula
        term1 = 1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(self.cov) ** 0.5)

        # To ensure more accurate matrix inversion, we can use the pinv (pseudo-inverse) function
        inv_cov = np.linalg.pinv(self.cov)
        term2 = np.exp(-0.5 * (x - self.mean).T @ inv_cov @ (x - self.mean))

        return float(term1 * term2)
