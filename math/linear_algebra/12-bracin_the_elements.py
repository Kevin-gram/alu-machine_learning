#!/usr/bin/env python3
""" implements the np_elementwise function
"""


def np_elementwise(mat1, mat2):
    """ performs elementwise operations on numpy arrays
        the result returned is a tuple containing the
        sum, difference, product and quotient of the arrays
    """
    sum = mat1 + mat2
    difference = mat1 - mat2
    product = mat1 * mat2
    quotient = mat1 / mat2

    return sum, difference, product, quotient
