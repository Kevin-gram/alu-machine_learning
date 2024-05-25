#!/usr/bin/env python3
"""
this module contains the function to calculate the minor of a matrix.
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


def minor(matrix):
    """
    Calculates the minor of a matrix.
    """
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]] or not all(len(matrix) == len(row) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    minor_matrix = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j+1:] for row in
                          (matrix[:i] + matrix[i+1:])]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)
    return minor_matrix
