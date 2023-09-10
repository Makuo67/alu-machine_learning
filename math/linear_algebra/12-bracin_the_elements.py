#!/usr/bin/env python3

"""Function that performs element-wise addition, subtraction, multiplication, and division"""


import numpy as np


def np_elementwise(mat1, mat2):
    """Performs some arithmetic"""
    if mat1.shape != mat2.shape:
        raise ValueError("Input matrices must have the same shape.")

    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return add, sub, mul, div