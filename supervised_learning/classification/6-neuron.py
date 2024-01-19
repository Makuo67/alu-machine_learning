#!/usr/bin/env python3
"""Neuron for Binary Classification"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Forward propagation"""
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """Cost function"""
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluate function"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Gradient descent"""
        m = X.shape[1]
        dz = A - Y
        dw = np.dot(X, dz.T) / m
        db = np.sum(dz) / m
        self.__W -= alpha * dw.T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")

        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
