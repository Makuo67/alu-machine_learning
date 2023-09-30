#!/usr/bin/env python3
"""Poisson distribution"""


class Poisson:
    """POisson distribution class"""

    def __init__(self, data=None, lambtha=1.):
        """Init"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """PMF function calc"""
        k = int(k)  # Convert k to an integer if it is not
        if k < 0:  # k is out of range
            return 0

        # Calculating factorial of k manually
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i

        # Calculating e^(-lambtha) manually
        exp_neg_lambtha = 1
        term = 1
        for i in range(1, 10):  # Summing up to 10 terms
            term *= -self.lambtha / i  # each term is x^i/i! for Taylor series of e^x
            exp_neg_lambtha += term

        return (exp_neg_lambtha * (self.lambtha ** k)) / factorial_k
