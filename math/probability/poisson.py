#!/usr/bin/env python3
"""Poisson distribution"""
import math


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
        """Pmf of poisson distribution"""
        k = int(k)  # Convert k to an integer if it is not
        if k < 0:  # k is out of range
            return 0
        return (math.exp(-self.lambtha) * (self.lambtha ** k)) / math.factorial(k)
