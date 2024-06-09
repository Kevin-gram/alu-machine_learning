#!/usr/bin/env python3
"""
This class represents a Binomial distribution
"""


class Binomial:
    """
    This class represents a Binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        function initializes the Binomial distribution
        data - list of the data to be used to estimate the distribution
        n - number of Bernoulli trials
        p - probability of a “success”
        """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1.0 - variance / mean
            self.n = round((mean / p))
            self.p = float(mean / self.n)

    def factorial(self, k):
        """ Find factorial of a number """
        result = 1
        for i in range(1, k+1):
            result *= i
        return result

    def pmf(self, k):
        """
        Calculates the PMF for a given number of "successes" k
        k - The number of "successes"
        Returns the PMF value for k.
        """
        if k < 0:
            return 0
        k = int(k)
        n_f = self.factorial(self.n)
        k_f = self.factorial(k)
        nk_f = self.factorial(self.n - k)
        pk = self.p ** k
        pmf = (n_f / (k_f * (nk_f))) * pk * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        '''calculate CDF for a given number of "successes" k'''
        if k < 0:
            return 0
        k = int(k)
        cdf = sum(self.pmf(i) for i in range(k + 1))
        return cdf
