#!/usr/bin/env python3
"""
This class represents a Normal distribution
"""


class Normal:
    """
    This class represents a Normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        function initializes the normal distribution
        data - list of the data to be used to estimate the distribution
        mean - mean of the distribution
        stddev - standard deviation
        """
        # Updated values for pi and Euler's number
        self.PI = 3.141592653589793
        self.E = 2.718281828459045

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
            self.mean = float(mean)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            squared_diff = [(x - self.mean) ** 2 for x in data]
            # Use the population standard deviation formula
            self.stddev = (sum(squared_diff) / len(data)) ** 0.5

    def z_score(self, x):
        """
        calculates z-score of a given x-value
        x is the value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        calculates x-value of the given z-score
        z - z-score
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        calculates the value of the PDF for a given x-value
        x - the x-value
        # """
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        pdf_value = (1 / (self.stddev * ((2 * self.PI) ** 0.5))) * \
            (self.E ** exponent)
        return pdf_value

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value
        x - the x-value
        """
        erf = (x - self.mean) / (self.stddev * (2 ** 0.5))
        cdf = 0.5 * (1 + (self.erf(erf)))
        return cdf

    def erf(self, x):
        """ Calculates the error function
        """
        return (2 / (self.PI ** 0.5)) * (x - (x ** 3) / 3 + (x ** 5) / 10
                                         - (x ** 7) / 42 + (x ** 9) / 216)
