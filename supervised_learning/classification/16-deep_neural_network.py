#!/usr/bin/env python3
"""Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network Class"""

    def __init__(self, nx, layers):
        """Constructor for DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Initialize layers
        layer_sizes = [nx] + layers
        for l in range(1, self.L + 1):
            layer_size = layer_sizes[l]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            weight_key = 'W' + str(l)
            bias_key = 'b' + str(l)
            self.weights[weight_key] = np.random.randn(
                layer_size, layer_sizes[l - 1]) * np.sqrt(
                    2 / layer_sizes[l - 1])
            self.weights[bias_key] = np.zeros((layer_size, 1))
