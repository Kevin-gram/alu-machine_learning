#!/usr/bin/env python3

'''This script calculates the Multivers dist and PDF'''


import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Class constructor.

        Parameters:
        data (numpy.ndarray): A 2D array of shape (d, n)
            n (int): The number of data points
            d (int): The number of dimensions in each data point

        Raises:
        TypeError: If data is not a 2D numpy.ndarray
        ValueError: If n is less than 2
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean of the data set
        self.mean = np.mean(data, axis=1).reshape(d, 1)

        # Center the data by subtracting the mean
        data_centered = data - self.mean

        # Calculate the covariance matrix
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Parameters:
        x (numpy.ndarray): A 2D array of shape (d, 1)
        Returns:
        float: The value of the PDF

        Raises:
        TypeError: If x is not a numpy.ndarray
        ValueError: If x is not of shape (d, 1)
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Calculate the PDF
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        norm_factor = 1 / np.sqrt((2 * np.pi) ** d * det_cov)
        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv_cov), x_centered)

        pdf_value = norm_factor * np.exp(exponent)

        return float(pdf_value)
