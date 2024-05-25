#!/usr/bin/env python3
"""
This module contains the function to calculate the determinant of a matrix.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.
    """
    if matrix == [[]]:
        return 1

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    size = len(matrix)
    answer = 0
    for j in range(size):
        sign = (-1)**j
        sub_matrix = [row[:j] + row[j+1:] for row in matrix[1:]]
        answer += sign * matrix[0][j] * determinant(sub_matrix)
    return answer
