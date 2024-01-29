#!/usr/bin/env python3
"""Calculates the cost of a neural network with L2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates cost of a neural network with L2 regularization.

    Args:
    - cost: the cost of the network without L2 regularization
    - lambtha: the regularization parameter
    - weights: a dictionary of the weights and biases (
        numpy.ndarrays) of the neural network
    - L: the number of layers in the neural network
    - m: the number of data points used

    Returns:
    - The cost of the network accounting for L2 regularization
    """

    l2_regularization_term = 0

    for layer in range(1, L + 1):
        W_key = 'W' + str(layer)
        if W_key in weights:
            l2_regularization_term += np.sum(
                np.square(weights[W_key]))

    l2_regularization_term *= (lambtha / (2 * m))

    cost_with_l2_regularization = cost + l2_regularization_term

    return cost_with_l2_regularization
