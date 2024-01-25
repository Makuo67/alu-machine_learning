#!/usr/bin/env python3
"""Placeholders for Neural network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates the placeholders for the neural network.

    Arguments:
    nx: the number of feature columns in our data
    classes: the number of classes in our classifier

    Returns:
    x: placeholder for the input data
    y: placeholder for the one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y
