#!/usr/bin/env python3
""" implements the add array method
"""


def add_arrays(arr1, arr2):
    """ adds two arrays elementwise
    """
    if len(arr1) != len(arr2):
        return None

    result = []

    for num1, num2 in zip(arr1, arr2):
        result.append(num1 + num2)

    return result
