#!/usr/bin/env python3

"""
This module contains a function that
calculates the intersection of obtaining
this data with the various hypothetical
probabilities
"""

import numpy as np
likelihood = __import__('0-likelihood').likelihood


def intersection(x, n, P, Pr):
    """
    x is the number of patients that develop side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P

    Returns: a 1D numpy.ndarray containing the intersection of
             obtaining x and n with each probability in P, respectively

    formula for intersection:
    P(E|H) = P(E and H) / P(H)
    intersection = P(E and H)
    P(E and H) = P(E|H) * P(H)
    P(H) = Pr
    P(E|H) = likelihood
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is "
                         "greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not isinstance(Pr, np.ndarray) or len(Pr.shape) != 1:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(Pr > 1) or np.any(Pr < 0):
        # raise ValueError("All values in Pr must be in the range [0, 1]")
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # P(E|H) = likelihood
    likelihoods = likelihood(x, n, P)
    # P(H) = Pr
    # intersection = P(E and H) = P(E|H) * P(H)
    intersection = likelihoods * Pr
    return intersection
