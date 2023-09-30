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

    def factorial(self, n):
        """Factorial"""
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n - 1)

    def exp(self, x, terms=10):
        """Calculating exponential using the Taylor Series expansion"""
        result = 1.0  # 0th term
        power = 1.0  # x^0 = 1 initially
        factorial = 1.0  # 0! = 1 initially
        for i in range(1, terms):
            power *= x  # x^i
            factorial *= i  # i!
            result += power / factorial  # Summing the series
        return result

    def pmf(self, k):
        """PMF"""
        k = int(k)  # Convert k to an integer if it is not
        if k < 0:  # k is out of range
            return 0
        # Calculating PMF using manually implemented exp and factorial functions
        return (self.exp(-self.lambtha) * (self.lambtha ** k)) / self.factorial(k)
