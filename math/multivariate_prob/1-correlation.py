#!/usr/bin/env python3

'''This is a script that calculates the correlation'''


import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.

    Parameters:
    C (numpy.ndarray): A 2D array of shape (d, d)
        d (int): The number of dimensions

    Returns:
    numpy.ndarray: A 2D array of shape (d, d) containing the correlation matrix

    Raises:
    TypeError: If C is not a numpy.ndarray
    ValueError: If C does not have shape (d, d)
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]

    # Calculate the standard deviations
    std_devs = np.sqrt(np.diag(C))

    # Create the correlation matrix
    correlation_matrix = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            correlation_matrix[i, j] = C[i, j] / (std_devs[i] * std_devs[j])

    return correlation_matrix
