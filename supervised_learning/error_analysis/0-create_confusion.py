#!/usr/bin/env python3
"""
Creating a Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.

    Arguments:
        - labels: a one-hot numpy.ndarray of shape (m, classes),
          representing the correct labels for each data point.
            * m: the number of data points.
            * classes: the number of classes.
        - logits: a one-hot numpy.ndarray of shape (m, classes),
          representing the predicted labels.

    Returns:
        A confusion matrix as a numpy.ndarray of shape (classes, classes),
        where rows represent the true labels and columns represent the
        predicted labels.
    """
    return np.dot(labels.T, logits)

