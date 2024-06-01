#!/usr/bin/env python3
'''
    The function below calculates
    the integral of a polynomial
'''


def poly_integral(poly, C=0):
    """
    calculates the integral of the given polynomial
    """
    if type(poly) is not list or len(poly) < 1:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    for coefficient in poly:
        if type(coefficient) is not int and type(coefficient) is not float:
            return None
    if type(C) is float and C.is_integer():
        C = int(C)
    integral = [C]
    for power, coefficient in enumerate(poly):
        if (coefficient % (power + 1)) is 0:
            new_coefficient = coefficient // (power + 1)
        else:
            new_coefficient = coefficient / (power + 1)
        integral.append(new_coefficient)
    while integral[-1] is 0 and len(integral) > 1:
        integral = integral[:-1]
    return integral
