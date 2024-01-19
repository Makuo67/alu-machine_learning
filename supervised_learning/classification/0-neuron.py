#!/usr/bin/env python3
"""This module is of a single perfoming classification"""
import numpy as np


class Neuron:
    """class that defines a single neuron performing classification"""

    def __init__(self, nx):
        """class constructor
        """
        # nx - no. of input features to the neuron
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Mean and standard deviation
        mean = 0
        std_dev = 1

        # Generate a random number from a normal distribution
        self.W = np.random.normal(0, 1, (nx, 1))

        # b - bias of the neuron
        self.b = 0

        # A - activated output of the neuron(Prediction)
        self.A = 0
