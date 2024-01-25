#!/usr/bin/env python3
"""Calculates the softmax cross-entropy loss of a prediction"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.

    Arguments:
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network’s predictions

    Returns:
    tensor containing the loss of the prediction
    """
    # Calculate softmax cross-entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=y_pred))

    return loss
