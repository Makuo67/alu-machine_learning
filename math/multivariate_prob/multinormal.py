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


# If you'd like to test the class:
# if __name__ == '__main__':
#     np.random.seed(0)
#     data = np.random.multivariate_normal(
#         [12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
#     mn = MultiNormal(data)
#     print(mn.mean)
#     print(mn.cov)
