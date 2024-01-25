#!/usr/bin/env python3
"""Accuracy Calculation"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Arguments:
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network’s predictions

    Returns:
    tensor containing the decimal accuracy of the prediction
    """
    # Compare predicted labels with true labels
    correct_predictions = tf.equal(
        tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))

    # Convert boolean values to floating-point values (0 or 1)
    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32))

    return accuracy
