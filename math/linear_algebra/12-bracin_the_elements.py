#!/usr/bin/env python3

"""Performs addition, subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """Performs some arithmetic"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div
