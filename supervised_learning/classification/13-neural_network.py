#!/usr/bin/env python3
"""Neural Network Binary Classification"""
import numpy as np


class NeuralNetwork:
    """Neural Network class"""

    def __init__(self, nx, nodes):
        """Constructor NeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = np.zeros((1, 1))
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network"""
        Z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.W2, self.__A1) + self.b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A1, self.__A2

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network"""
        m = Y.shape[1]

        # Calculate gradients of the output layer
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Calculate gradients of the hidden layer
        dZ1 = np.dot(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update the weights and biases
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
