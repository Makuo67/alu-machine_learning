#!/usr/bin/env python3
"""Layers for TensorFlow"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Arguments:
    x: placeholder for the input data
    layer_sizes: the number of nodes in each layer of the network
    activations: the activation functions for each layer of the network

    Returns:
    the prediction of the network in tensor form
    """
    # Ensure the number of layers and activations match
    assert len(layer_sizes) == len(
        activations), "Number of layers and activations must be the same"

    # Initialize the input tensor
    prev = x

    # Create each layer in the network
    for i in range(len(layer_sizes)):
        # Use the create_layer function to create each layer
        prev = create_layer(prev, layer_sizes[i], activations[i])

    return prev
