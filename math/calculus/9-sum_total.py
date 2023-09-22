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
    # Recursive case: sum i^2 from 1 to n
    return (n**2 + summation_i_squared(n-1))

# n = 5
# print(summation_i_squared(n))       
