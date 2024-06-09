#!/usr/bin/env python3
"""
This class represents a poisson distribution
"""


class Poisson:
    """
    This class represents a poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        This function initializes the poisson distribution
        and calculates lambtha if data is given
        data - list of the data to be used to estimate the distribution
        lambtha - expected number of occurences in a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for i in range(1, k + 1):
            factorial = factorial * i
        result = self.lambtha ** k * 2.7182818285 ** (-self.lambtha)
        return result / factorial

    def cdf(self, k):
        """
        Calculate the value of the CDF for a given number of “successes”
        k = "successes"
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        result = 0
        for i in range(k + 1):
            result = result + self.pmf(i)
        return result
