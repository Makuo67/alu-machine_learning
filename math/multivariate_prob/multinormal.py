#!/usr/bin/env python3
"""Multivariate Normal distribution"""


import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """Class constructor for MultiNormal

        Parameters:
        - data: numpy.ndarray of shape (d, n) containing the dataset
        """

        # Check if data is a 2D numpy.ndarray
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        # Check if n is less than 2
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean of the dataset
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate the covariance matrix
        deviation = data - self.mean
        self.cov = np.dot(deviation, deviation.T) / (n - 1)

    def pdf(self, x):
        """ Calculate the PDF at a data point """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Compute the PDF of multivariate normal distribution
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)

        term1 = 1 / (np.sqrt((2 * np.pi)**d * det_cov))
        diff = x - self.mean
        term2 = np.exp(-0.5 * np.dot(diff.T, np.dot(inv_cov, diff)))

        return term1 * term2[0, 0]
