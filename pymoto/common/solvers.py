import numpy as np
import scipy.sparse as sps
try:
    import cvxopt
    _has_cvxopt = True
except ImportError:
    _has_cvxopt = False


def is_cvxopt_spmatrix(A):
    """ Checks if the argument is a cvxopt sparse matrix """
    return isinstance(A, cvxopt.spmatrix) if _has_cvxopt else False


def matrix_is_complex(A):
    """ Checks if the matrix is complex """
    if is_cvxopt_spmatrix(A):
        return A.typecode == 'z'
    else:
        return np.iscomplexobj(A)


def matrix_is_diagonal(A):
    """ Checks if the matrix is diagonal"""
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
    """ Checks whether a matrix is numerically symmetric """
    if sps.issparse(A):
        return np.allclose((A-A.T).data, 0)
    elif is_cvxopt_spmatrix(A):
        return np.isclose(max(abs(A-A.T)), 0.0)
    else:
        return np.allclose(A, A.T)


def matrix_is_hermitian(A):
    """ Checks whether a matrix is numerically Hermitian """
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
    """ Base class of all linear solvers """

    defined = True  # Flag if the solver is able to run, e.g. false if some dependent library is not available
    _err_msg = ""

    def __init__(self, A=None):
        if A is not None:
            self.update(A)

    def update(self, A):
        """ Updates with a new matrix of the same structure

        Args:
            A: The updated matrix

        Returns:
            self
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def solve(self, rhs):
        """ Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}`

        Args:
            rhs: Right hand side :math:`\mathbf{b}`

        Returns:
            Solution vector :math:`\mathbf{x}`
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def adjoint(self, rhs):
        """ Solves the adjoint linear system of equations

        The system of equations is :math:`\mathbf{A}^\\text{H} \mathbf{x} = \mathbf{b}` (conjugate transpose) in case of
        complex matrix or :math:`\mathbf{A}^\\text{T} \mathbf{x} = \mathbf{b}` for a real-valued matrix.

        Args:
            rhs: Right hand side :math:`\mathbf{b}`

        Returns:
            Solution vector :math:`\mathbf{x}`
        """
        return self.solve(rhs.conj()).conj()

    @staticmethod
    def residual(A, x, b):
        """ Calculates the (relative) residual of the linear system of equations

        The residual is calculated as
        :math:`r = \\frac{\left| \mathbf{A} \mathbf{x} - \mathbf{b} \\right|}{\left| \mathbf{b} \\right|}`

        Args:
            A: The matrix
            x: Solution vector
            b: Right-hand side

        Returns:
            Residual value
        """
        return np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)


class LDAWrapper(LinearSolver):
    """ Linear dependency aware solver (LDAS)

    This solver uses previous solutions of the system :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` to reduce computational
    effort. In case the solution :math:`\mathbf{x}` is linearly dependent on the previous solutions, the solution
    will be nearly free of cost.

    Args:
        solver: The internal solver to be used
        tol (optional): Residual tolerance above which the internal solver is used to add a new solution vector.
        A (optional): The matrix :math:`\mathbf{A}`

    References:
        Koppen, van der Kolk, van den Boom, Langelaar (2022).
        `doi: 10.1007/s00158-022-03378-8 <https://doi.org/10.1007/s00158-022-03378-8>`_
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
        """ Clear the internal stored solution vectors and update the internal ``solver`` """
        self.A = A
        self.x_stored.clear()
        self.b_stored.clear()
        self.solver.update(A)

    def solve(self, rhs):
        """ Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by performing a modified
        Gram-Schmidt over the previously calculated solutions :math:`\mathbf{U}` and corresponding right-hand-sides
        :math:`\mathbf{F}`. This is used to construct an approximate solution
        :math:`\\tilde{\mathbf{x}} = \sum_k \\alpha_k \mathbf{u}_k` in the subspace of :math:`\mathbf{U}`.
        If the residual of :math:`\mathbf{A} \\tilde{\mathbf{x}} = \mathbf{b}` is above the tolerance, a new solution
        :math:`\mathbf{u}_{k+1}` will be added to the database such that
        :math:`\mathbf{x} = \\tilde{\mathbf{x}}+\mathbf{u}_{k+1}` is the solution to the system
        :math:`\mathbf{A} \mathbf{x} = \mathbf{b}`.
        """
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
