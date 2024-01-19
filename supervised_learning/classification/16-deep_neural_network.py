#!/usr/bin/env python3
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

        # Check if layers are positive integers
        if any(not isinstance(layer, int) or layer <= 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # He et al. weight initialization
        for layer in range(1, self.L + 1):
            layer_size = layers[layer - 1]
            prev_layer_size = nx if layer == 1 else layers[layer - 2]
            self.weights[f'W{layer}'] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            self.weights[f'b{layer}'] = np.zeros((layer_size, 1))
