#!/usr/bin/env python3
"""Updates the weights and biases of a neural network using gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Args:
    - Y: a one-hot numpy.ndarray of shape (classes, m)
    - weights:dictionary of the weights and biases of the network
    - cache: dictionary of the outputs of each layer of the network
    - alpha: the learning rate
    - lambtha: the L2 regularization parameter
    - L: the number of layers of the network

    Returns:
    - None (updates the weights and biases in place)
    """

    m = Y.shape[1]

    for layer in range(L, 0, -1):
        A_key = 'A' + str(layer)
        A_prev_key = 'A' + str(layer - 1)
        W_key = 'W' + str(layer)
        b_key = 'b' + str(layer)

        if layer == L:
            dZ = cache[A_key] - Y
        else:
            dZ = np.dot(weights['W' + str(layer + 1)].T, dZ) * (
                1 - np.power(cache[A_key], 2))

        dW = np.dot(dZ, cache[A_prev_key].T) / m + (
            lambtha / m) * weights[W_key]
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db
