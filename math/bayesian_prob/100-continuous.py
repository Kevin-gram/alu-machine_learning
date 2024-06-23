#!/usr/bin/env python3

"""
This module contains a function that
calculates the posterior probability
that the probability of developing
severe side effects falls within a
specific range given the data
"""

from scipy import special


def posterior(x, n, p1, p2):
    """
    calculates the posterior probability

    x--> number of patients that develop side effects
    n--> total number of patients observed
    p1--> lower bound on the range
    p2--> upper bound on the range

    returns: the posterior probability that p is within
                the range [p1, p2] given x and n
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError(
            "n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError(
            "p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError(
            "p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError(
            "p2 must be greater than p1")

    # Calculate the posterior probability using the Beta distribution
    posterior_prob = special.betainc(
        x + 1, n - x + 1, p2) - special.betainc(x + 1, n - x + 1, p1)

    return posterior_prob
