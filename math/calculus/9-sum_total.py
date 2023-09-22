#!/usr/bin/env python3
"""Sigma"""


def summation_i_squared(n):
    """Summation"""
    # Check if n is a valid number (integer)
    if not isinstance(n, int) or n < 1:
        return None

    # Base case: when n reaches 1, return 1^2 = 1
    if n == 1:
        return 1
    # Sum of natural numbers
    return (n*(n+1)*(2*n+1))/6
