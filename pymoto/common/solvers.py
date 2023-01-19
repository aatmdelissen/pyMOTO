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


class LinearSolver:
    defined = True
    _err_msg = ""

    def __init__(self, A=None):
        if A is not None:
            self.update(A)

    def update(self, A):
        """ Updates with a new matrix of the same structure
        :param A: The new matrix
        :return: self
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def solve(self, rhs):
        """ Solves A x = rhs
        :param rhs: Right hand side
        :return: Solution vector x
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def adjoint(self, rhs):
        """Solves A^H x = rhs in case of complex matrix or A^T x = rhs for a real-valued matrix
        :param rhs: Right hand side
        :return: Solution vector x
        """
        return self.solve(rhs.conj()).conj()
        # raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    @staticmethod
    def residual(A, x, b):
        """ Calculates the residual || A x - b || / || b ||
        :param A: Matrix
        :param x: Solution
        :param b: Right-hand side
        :return: Residual value
        """
        return np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)


class LDAWrapper(LinearSolver):
    """ Linear dependency aware solver (LDAS)
    Koppen, van der Kolk, van den Boom (2022) https://doi.org/10.1007/s00158-022-03378-8
    """
    def __init__(self, solver: LinearSolver, tol=1e-8, A=None):
        self.solver = solver
        self.tol = tol
        self.x_stored = []
        self.b_stored = []
        self.A = None
        self._did_solve = False  # For debugging purposes
        self._last_rtol = 0.
        super().__init__(A)

    def update(self, A):
        self.A = A
        self.x_stored.clear()
        self.b_stored.clear()
        self.solver.update(A)

    def solve(self, rhs):
        rhs_loc = rhs.copy()
        sol = np.zeros_like(rhs_loc)

        # Check linear dependencies in the rhs using modified Gram-Schmidt
        for (x, b) in zip(self.x_stored, self.b_stored):
            alpha = rhs_loc @ b / (b @ b)
            rhs_loc -= alpha * b
            sol += alpha * x

        # Check tolerance
        self._last_rtol = self.residual(self.A, sol, rhs)
        if self._last_rtol > self.tol:
            # Calculate a new solution
            xnew = self.solver.solve(rhs_loc)
            self.x_stored.append(xnew)
            self.b_stored.append(rhs_loc)
            sol += xnew
            self._did_solve = True
        else:
            self._did_solve = False

        return sol
