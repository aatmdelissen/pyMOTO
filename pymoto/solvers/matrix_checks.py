import numpy as np
import scipy.sparse as sps

try:
    import cvxopt

    _has_cvxopt = True
except ImportError:
    _has_cvxopt = False


def is_cvxopt_spmatrix(A):
    """Checks if the argument is a cvxopt sparse matrix"""
    return isinstance(A, cvxopt.spmatrix) if _has_cvxopt else False


def matrix_is_sparse(A):
    return sps.issparse(A)


def matrix_is_complex(A):
    """Checks if the matrix is complex"""
    if is_cvxopt_spmatrix(A):
        return A.typecode == "z"
    else:
        return np.iscomplexobj(A)


def matrix_is_diagonal(A):
    """Checks if the matrix is diagonal"""
    if matrix_is_sparse(A):
        if isinstance(A, sps.dia_matrix):
            return len(A.offsets) == 1 and A.offsets[0] == 0
        else:
            return np.allclose((A - sps.spdiags(A.diagonal(), 0, *A.shape)).data, 0.0)
    elif is_cvxopt_spmatrix(A):
        return max(abs(A.I - A.J)) == 0
    else:
        return np.allclose(A, np.diag(np.diag(A)))


def matrix_is_symmetric(A):
    """Checks whether a matrix is numerically symmetric"""
    if matrix_is_sparse(A):
        return np.allclose((A - A.T).data, 0)
    elif is_cvxopt_spmatrix(A):
        return np.isclose(max(abs(A - A.T)), 0.0)
    else:
        return np.allclose(A, A.T)


def matrix_is_hermitian(A):
    """Checks whether a matrix is numerically Hermitian"""
    if matrix_is_complex(A):
        if matrix_is_sparse(A):
            return np.allclose((A - A.T.conj()).data, 0)
        elif is_cvxopt_spmatrix(A):
            return np.isclose(max(abs(A - A.ctrans())), 0.0)
        else:
            return np.allclose(A, A.T.conj())
    else:
        return matrix_is_symmetric(A)


def matrix_is_positive_definite(A):
    """Check if the matrix is positive definite.

    By testing for strictly diagonally dominant matrix. 
    Diagonal dominant matrices ==> positive definite (but the reverse is not always true)
    
    https://math.stackexchange.com/questions/87528/a-practical-way-to-check-if-a-matrix-is-positive-definite
    """
    # The hermitian/symmetric part of the matrix determines positive definiteness
    Aherm = (A + A.conj().T)/2

    # If any of the diagonal is negative or complex, the matrix is not positive definite
    if Aherm.diagonal().real.min() < 0 or np.abs(np.angle(Aherm.diagonal())).max() > 1e-15:
        return False
    
    Adiag = np.diag(Aherm)
    
    # Test with Gershgorin circle theorem
    row_sum = np.sum(np.abs(Aherm), axis=1) - np.abs(Adiag)
    if np.all(Adiag > row_sum):
        return True

    # Cannot determine positive-definiteness
    return None
    
        