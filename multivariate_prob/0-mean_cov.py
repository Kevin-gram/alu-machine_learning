#!/usr/bin/env python3

'''This script finads the mean of a covariance '''


import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
    X (numpy.ndarray): A 2D array of shape (n, d) containing the data set
        n (int): The number of data points
        d (int): The number of dimensions in each data point

    Returns:
    mean (numpy.ndarray): A 1D array of shape (1, d)
    cov (numpy.ndarray): A 2D array of shape (d, d)

    Raises:
    TypeError: If X is not a 2D numpy.ndarray
    ValueError: If n is less than 2
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean of the data set
    mean = np.mean(X, axis=0).reshape(1, d)

    # Center the data by subtracting the mean
    X_centered = X - mean

    # Calculate the covariance matrix
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
