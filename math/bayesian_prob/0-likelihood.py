#!/usr/bin/env python3
"""Bayesian Probability"""


import numpy as np


def likelihood(x, n, P):
    """Check for valid input parameters"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if any(val < 0 or val > 1 for val in P):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the likelihood for each probability in P using the binomial probability mass function
    likelihoods = np.array(
        [np.math.comb(n, x) * p**x * (1 - p)**(n - x) for p in P])

    return likelihoods
