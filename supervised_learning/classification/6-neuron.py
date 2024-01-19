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

    @property
    def b(self):
        """getter function"""
        return self.__b

    @property
    def A(self):
        """getter function"""
        return self.__A

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

    def evaluate(self, X, Y):
        """evaluating neuron's predictions
        X - np.ndarray of shape (nx, m) contains input data
        Y - np.ndarray of shape (1, m) contains correct labels for input data
        """

        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)
        accuracy = np.mean(predictions == Y)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculates one pass of gradient descent on the neuron
        X - np.ndarray of shape (nx, m) contains input data
        Y - np.ndarray of shape (1, m) contains correct labels for input data
        A - contains activated output of the neuron
        alpha - learning rate
        """
        m = X.shape[1]  # examples
        dw = np.dot(X, (A - Y).T) / m
        db = np.sum(A - Y) / m

        # updating weights and bias
        self.__W = self.__W - alpha*dw
        self.__b = self.__b - alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train the neuron
        X - has input data
        Y - has correct labels for input data
        iterations - no. of iterations to train over
        alpha - learning rate
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha=0.05)
            # if i % 1000 == 0:
            #     cost = self.cost(Y, A)
            #     print(f"cost after iteration {i}: {cost}")

        return self.evaluate(X, Y)
