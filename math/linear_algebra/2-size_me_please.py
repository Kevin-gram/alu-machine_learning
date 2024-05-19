#!/usr/bin/env python3
'''
    calculation of  the shape of a matrix
'''


def matrix_shape(matrix):
    '''
         shape of a matrix
    '''
    mat_shape = []
    while isinstance(matrix, list):
        mat_shape.append(len(matrix))
        matrix = matrix[0]
    return mat_shapei
