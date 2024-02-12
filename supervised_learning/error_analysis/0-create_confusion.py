#!/usr/bin/env python3
"""Error Analysis"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Args:
    - labels: one-hot numpy.ndarray of shape (m, classes) containing the correct labels for each data point
    - logits: one-hot numpy.ndarray of shape (m, classes) containing the predicted labels

    Returns:
    - confusion: numpy.ndarray of shape (classes, classes) with row indices representing the correct labels
                 and column indices representing the predicted labels
    """
    # Convert one-hot encoded labels and logits to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # Calculate the confusion matrix
    confusion = np.zeros((labels.shape[1], labels.shape[1]), dtype=np.int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion[true_label, predicted_label] += 1

    return confusion
