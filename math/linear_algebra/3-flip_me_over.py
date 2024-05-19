#!/usr/bin/env python3
'''
a function that returns the transpose of a 2D matrix:
'''


def matrix_transpose(matrix):
    '''
        returns the transpose of a 2D matrix, matrix:
    '''
    result = [
        [matrix[j][i] for j in range(len(matrix))]
        for i in range(len(matrix[0]))
    ]
    return result
