#!/usr/bin/env python3
"""Derivative of a polynomial"""


def poly_derivative(poly):
    """Derivative"""

    derivative = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    for i in range(len(poly)-1, 0, -1):

        derivative.append(poly[i]*i)
    return derivative[::-1]
