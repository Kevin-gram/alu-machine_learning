#!/usr/bin/env python3
""" implements the cat_matrices2D function
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenates two matrices along a specified axis
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        return mat1.copy() + mat2.copy()
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        result = []

        for row1, row2 in zip(mat1, mat2):
            combined_row = row1 + row2
            result.append(combined_row)

        return result
