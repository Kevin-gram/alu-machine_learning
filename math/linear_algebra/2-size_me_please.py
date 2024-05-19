#!/usr/bin/env python3
'''
    This script has a function that calculates the shape of a matrix
'''


def matrix_shape(matrix):
    '''
        Calculates the shape of a matrix
    '''
    mat_shape = []
    while isinstance(matrix, list):
        mat_shape.append(len(matrix))
        matrix = matrix[0]
    return mat_shape
