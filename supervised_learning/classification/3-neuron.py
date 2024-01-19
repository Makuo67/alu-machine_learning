#!/usr/bin/env python3
"""This module is of a binary classification"""
import numpy as np


class Neuron:
    """class that defines a single neuron perfoming binary classification"""

    def __init__(self, nx):
        """ class construtor"""

        # nx - no. of input features to the neuron
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        # w - weights vector of the neuron
        self.__W = np.random.normal(0, 1, (nx, 1))

        # Initialize the bias the neuron
        self.__b = 0

        # Initialize the activated output of the neuron (Prediction)
        self.__A = 0

    # getter function
    @property
    def W(self):
        """getter function"""
        return self.__W

    # # setter function
    # @W.setter
    # def W(self, value):
    #     """setter function"""
    #     self.__W = value

    @property
    def b(self):
        """getter function"""
        return self.__b

    # # setter function
    # @b.setter
    # def b(self, value):
    #     """setter function"""
    #     self.__b = value

    @property
    def A(self):
        """getter function"""
        return self.__A

    # # setter function
    # @A.setter
    # def A(self, value):
    #     """setter function"""
    #     self.__A = value

    def forward_prop(self, X):
        """calculating forward propagation of the neuron"""
        # X - a np.ndarray of shape (nx, m)
        # nx - input features to the neuron
        # m - no. of examples

        # weighted sum
        weighted_sum = np.dot(self.__W.T, X) + self.__b
        # applying activation function
        self.__A = 1/(1 + np.exp(-weighted_sum))

        return self.__A

    def cost(self, Y, A):
        """ calculates cost of the model using logistic regression
        Y - contains correct labels of input data
        A - contains activated output of the neuron"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost
