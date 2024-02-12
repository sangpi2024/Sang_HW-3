import numpy as np
import scipy.linalg as la

def is_symmetric(A):
    """
    Check if a matrix A is symmetric.

    Args:
        A (np.ndarray): A square numpy array representing the matrix.

    Returns:
        bool: True if A is symmetric, False otherwise.
    """
    # np.allclose checks if all elements in A and its transpose are close enough to be considered equal.
    return np.allclose(A, A.T)

def is_positive_definite(A):
    """
    Check if a matrix A is positive definite.

    Args:
        A (np.ndarray): A square numpy array representing the matrix.

    Returns:
        bool: True if A is positive definite, False otherwise.
    """
    try:
        _ = la.cholesky(A)   # Attempt to perform a Cholesky decomposition, which only succeeds for positive definite matrices.
        return True
    except la.LinAlgError:   # If Cholesky decomposition fails, the matrix is not positive definite.
        return False

def cholesky_solve(A, b):
    """
    Solve the linear system Ax = b using Cholesky decomposition.

    Args:
        A (np.ndarray): A symmetric, positive definite matrix.
        b (np.ndarray): The right-hand side vector.

    Returns:
        np.ndarray: Solution vector x.
    """
    L = la.cholesky(A, lower=True)              # Perform Cholesky decomposition to get L, where A = LL^T.
    y = la.solve_triangular(L, b, lower=True)   # Solve Ly = b for y using forward substitution.
    x = la.solve_triangular(L.T, y)             # Solve L^Tx = y for x using back substitution.
    return x

def doolittle_solve(A, b):
    """
    Solve the linear system Ax = b using LU decomposition (Doolittle's method).

    Args:
        A (np.ndarray): A square matrix.
        b (np.ndarray): The right-hand side vector.

    Returns:
        np.ndarray: Solution vector x.
    """

    P, L, U = la.lu(A)                                          # Perform LU decomposition where A = PLU. P is the permutation matrix.
    y = la.solve_triangular(L, np.dot(P.T, b), lower=True)      # Solve Ly = Pb for y using forward substitution. P.T is the transpose of the permutation matrix.
    x = la.solve_triangular(U, y)                               # Solve Ux = y for x using back substitution.
    return x

# Define the matrix and vector
A1 = np.array([[1, -1, 3, 2],
               [-1, 5, -5, -2],
               [3, -5, 19, 3],
               [2, -2, 3, 21]])

b1 = np.array([15, -35, 94, 1])

# Check matrix properties and solve
if is_symmetric(A1) and is_positive_definite(A1):
    solution = cholesky_solve(A1, b1)
    print("Solution using Cholesky:", solution)
else:
    solution = doolittle_solve(A1, b1)
    print("Solution using Doolittle (LU Factorization):", solution)
