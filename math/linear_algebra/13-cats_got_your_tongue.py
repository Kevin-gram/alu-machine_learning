#!/usr/bin/env python3
'''
    This function def np_cat(mat1, mat2, axis=0)
    concatenates two matrices along a specific axis
'''


import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''
        Concatenate two arrays based on an axis
    '''
    return np.concatenate((mat1, mat2), axis=axis)
