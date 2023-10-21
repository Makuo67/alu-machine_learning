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
        """ Calculate the PDF at a data point """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Compute the PDF of multivariate normal distribution
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)
        mahalanobis = np.dot(
            np.dot((x - self.mean).T, cov_inv), (x - self.mean))

        term1 = 1 / (np.power((2 * np.pi), d/2) * np.sqrt(cov_det))
        term2 = np.exp(-0.5 * mahalanobis)

        return term1 * term2
