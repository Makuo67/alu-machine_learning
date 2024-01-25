#!/usr/bin/env python3
"""Training Operation"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates training operation for the network using gradient descent.

    Arguments:
    loss: the loss of the network’s prediction
    alpha: the learning rate

    Returns:
    an operation that trains the network using gradient descent
    """
    # Use gradient descent optimizer with the specified learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # Minimize the loss by updating the trainable variables
    train_op = optimizer.minimize(loss)

    return train_op
