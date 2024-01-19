#!/usr/bin/env python3
"""This module is of a single perfoming classification"""
import numpy as np


class Neuron:
    """Defining a single neuron performing classification"""

    def __init__(self, nx):
        """ constructor
        """
        # Check if nx is an integer and greater than 0
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Mean and standard deviation
        mean = 0
        std_dev = 1

        # Initialize weights, bias, and activated output
        self.W = np.random.normal(0, 1, (1, nx))

        # b - bias of the neuron
        self.b = 0

        # A - activated output of the neuron(Prediction)
        self.A = 0
