import numpy as np
import scipy.sparse as sps
try:
    import cvxopt
    _has_cvxopt = True
except ImportError:
    _has_cvxopt = False


def is_cvxopt_spmatrix(A):
    return isinstance(A, cvxopt.spmatrix) if _has_cvxopt else False

def matrix_is_complex(A):
    if is_cvxopt_spmatrix(A):
        return A.typecode == 'z'
    else:
        return np.iscomplexobj(A)

def matrix_is_diagonal(A):
    if sps.issparse(A):
        if isinstance(A, sps.dia_matrix):
            return len(A.offsets) == 1 and A.offsets[0] == 0
        else:
            return np.allclose((A - sps.spdiags(A.diagonal(), 0, *A.shape)).data, 0.0)
    elif is_cvxopt_spmatrix(A) :
        return max(abs(A.I - A.J)) == 0
    else:
        return np.allclose(A, np.diag(np.diag(A)))


def matrix_is_symmetric(A):
    """ Returns whether a matrix is numerically symmetric or not """
    if sps.issparse(A):
        return np.allclose((A-A.T).data, 0)
    elif is_cvxopt_spmatrix(A):
        return np.isclose(max(abs(A-A.T)), 0.0)
    else:
        return np.allclose(A, A.T)


def matrix_is_hermitian(A):
    if matrix_is_complex(A):
        if sps.issparse(A):
            return np.allclose((A-A.T.conj()).data, 0)
        elif is_cvxopt_spmatrix(A):
            return np.isclose(max(abs(A-A.ctrans())), 0.0)
        else:
            return np.allclose(A, A.T.conj())
    else:
        return matrix_is_symmetric(A)



