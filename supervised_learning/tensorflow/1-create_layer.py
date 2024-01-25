#!/usr/bin/env python3
"""Layers for TensorFlow"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Arguments:
    prev: the tensor output of the previous layer
    n: the number of nodes in the layer to create
    activation: the activation function for the layer

    Returns:
    tensor output of the layer
    """
    # Use He et al. initialization for the layer weights
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    # Layer with specified number of nodes, activation, and initialization
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer')

    # Apply the layer to the previous tensor
    output = layer(prev)

    return output
