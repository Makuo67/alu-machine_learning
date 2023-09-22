#!/usr/bin/env python3
"""Derivative of a polynomial"""


def poly_derivative(poly):
    """Derivative"""
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly):
        # Check if poly is a list of valid coefficients
        return None

    if len(poly) < 2:
        # A polynomial of degree less than 1 has a derivative of 0
        return [0]

    derivative = []
    for power, coeff in enumerate(poly[1:], start=1):
        # Calculate the derivative coefficient for each term
        derivative_coeff = coeff * power
        derivative.append(derivative_coeff)

    return derivative
