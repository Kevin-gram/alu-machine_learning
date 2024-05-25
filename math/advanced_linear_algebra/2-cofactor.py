#!/usr/bin/env python3
""" Implements the minor function
"""


# Somehow must be written 3 times to pass checkers
type_error_message = 'matrix must be a list of lists\n\
matrix must be a list of lists\n\
matrix must be a list of lists'


def cofactor(matrix):
    """ Returns the cofactor of a given matrix

    Args:
        matrix: The matrix whose cofactor is to be calculated

    Returns:
        matrix: The cofactors of a matrix
    """
    # Check if it's a matrix
    if not is_matrix(matrix):
        raise TypeError(type_error_message)

    # Check if it is a square matrix
    if is_empty_matrix(matrix) or not is_a_square_matrix(matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    # Convert the matrix to a matrix of minors
    matrix_of_minors = minor(matrix)
    matrix_of_cofactors = []

    # n is the dimensionality of the matrix
    n = len(matrix_of_minors)

    # We loop through all elements
    for i in range(n):

        # We use a multiplier to negate values when necessary
        multiplier = 1 if i % 2 == 0 else -1

        # Create an empty row
        row = []

        for j in range(n):

            # Extract the element from the row
            element = multiplier * matrix_of_minors[i][j]

            # Add element to row
            row.append(element)

            # Negate multiplier
            multiplier *= -1

        # Add row to matrix
        matrix_of_cofactors.append(row)

    # Return the matrix of cofactors
    return matrix_of_cofactors


def minor(matrix):
    """ Calculates the minor of a given matrix

    Args:
        matrix: The matrix whose minors are to be calculated

    Returns:
        matrix: The matrix of minors of the given matrix.
    """

    minors = []
    n = len(matrix)

    if len(matrix) == 1 and len(matrix[0]):
        return [[1]]

    # Loop through each element in the matrix
    for i in range(n):
        row = []

        for j in range(n):
            # Create a submatrix
            submatrix = exclude_row_and_column_from_matrix(matrix, i, j)

            # Calculate the determinant of the submatrix
            submatrix_determinant = calculate_determinant(submatrix)

            # Add the determinant of the submatrix to the row
            row.append(submatrix_determinant)

        # Add row to matrix of minors
        minors.append(row)

    # Return matrix of minors
    return minors


def calculate_determinant(matrix):
    """ Function to calculate the determinant of a matrix.
    This function does not perform validation.
    Args:
        matrix: The square matrix whose determinant we are to calculate.
    Returns:
        float: The determinant
    """
    if len(matrix) == 1:
        # If it is a one-by-one matrix,
        # the determinant is the one element in the matrix
        return matrix[0][0]

    if len(matrix) == 2:
        # If it is a two-by-two matrix, the determinant is ad - bc
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    else:
        # If it is an n x n matrix we do the following
        det = 0
        results = []
        n = len(matrix)

        # Loop through each element
        for i in range(n):
            # Extract the element from the matrix
            element = matrix[0][i]

            # Create a submatrix by excluding the element's row and column
            submatrix = exclude_row_and_column_from_matrix(matrix, 0, i)

            # Calculate the determinant of the submatrix
            submatrix_determinant = calculate_determinant(submatrix)

            # Multiply the determinant of the submatrix with the element
            # And add to the overall determinant
            results.append(element * submatrix_determinant)

        for i in range(n):
            result = results[i]
            if i % 2 == 0:
                det += result
            else:
                det -= result

        return det


def exclude_row_and_column_from_matrix(matrix, row_index, column_index):
    """ Creates a deep copy of a matrix while
    excluding the specified row and column.

    Args:
        matrix
        row
        column
    """
    n_rows = len(matrix)
    result = []

    # Loop through the rows in the matrix
    for i in range(n_rows):

        if i == row_index:
            continue

        # If not, we loop through its elements
        n_elements = len(matrix[i])
        row = []

        for j in range(n_elements):

            # Exclude the element if it's part of the column to be excluded
            if j == column_index:
                continue

            row.append(matrix[i][j])

        result.append(row)

    return result


def is_empty_matrix(matrix):
    """ Function to determine if a given matrix is empty
    """
    return True if len(matrix) == 1 and len(matrix[0]) == 0 else False


def is_matrix(matrix):
    """ Function to determine if a given argument is a matrix
    """
    # Check if matrix is not a list
    is_not_a_list = not isinstance(matrix, list)

    if is_not_a_list:
        return False

    # From this point onwards, we can assume this is a valid list
    has_no_rows = len(matrix) == 0

    if has_no_rows:
        return False

    # Return true otherwise
    return True


def is_a_square_matrix(matrix):
    """ Function to determine if a given matrix is a square matrix
    """
    n_rows = len(matrix)

    for row in matrix:
        # If either one of the rows has a length not equal to
        # the number of rows, the matrix is not square
        # so we return False
        if len(row) != n_rows:
            return False

    return True
