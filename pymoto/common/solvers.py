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
    elif is_cvxopt_spmatrix(A):
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
    """ Base class of all linear solvers

    Keyword Args:
        A (matrix): Optionally provide a matrix, which is used in :method:`update` right away.

    Attributes:
        defined (bool): Flag if the solver is able to run, e.g. false if some dependent library is not available
    """

    defined = True
    _err_msg = ""

    def __init__(self, A=None):
        if A is not None:
            self.update(A)

    def update(self, A):
        """ Updates with a new matrix of the same structure

        Args:
            A (matrix): The new matrix of size ``(N, N)``

        Returns:
            self
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}`

        Args:
            rhs: Right hand side :math:`\mathbf{b}` of shape ``(N)`` or ``(N, K)`` for multiple right-hand-sides

        Returns:
            Solution vector :math:`\mathbf{x}` of same shape as :math:`\mathbf{b}`
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def adjoint(self, rhs):
        r""" Solves the adjoint linear system of equations

        The system of equations is :math:`\mathbf{A}^\text{H} \mathbf{x} = \mathbf{b}` (conjugate transpose) in case of
        complex matrix or :math:`\mathbf{A}^\text{T} \mathbf{x} = \mathbf{b}` for a real-valued matrix.

        Args:
            rhs: Right hand side :math:`\mathbf{b}` of shape ``(N)`` or ``(N, K)`` for multiple right-hand-sides

        Returns:
            Solution vector :math:`\mathbf{x}` of same shape as :math:`\mathbf{b}`
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    @staticmethod
    def residual(A, x, b):
        r""" Calculates the (relative) residual of the linear system of equations

        The residual is calculated as
        :math:`r = \frac{\left| \mathbf{A} \mathbf{x} - \mathbf{b} \right|}{\left| \mathbf{b} \right|}`

        Args:
            A: The matrix
            x: Solution vector
            b: Right-hand side

        Returns:
            Residual value
        """
        return np.linalg.norm(A@x - b) / np.linalg.norm(b)


class LDAWrapper(LinearSolver):
    r""" Linear dependency aware solver (LDAS)

    This solver uses previous solutions of the system :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` to reduce computational
    effort. In case the solution :math:`\mathbf{x}` is linearly dependent on the previous solutions, the solution
    will be nearly free of cost.

    Args:
        solver: The internal solver to be used
        tol (optional): Residual tolerance above which the internal solver is used to add a new solution vector.
        A (optional): The matrix :math:`\mathbf{A}`

    References:

    Koppen, S., van der Kolk, M., van den Boom, S., & Langelaar, M. (2022).
    Efficient computation of states and sensitivities for compound structural optimisation problems using a Linear Dependency Aware Solver (LDAS).
    Structural and Multidisciplinary Optimization, 65(9), 273.
    DOI: 10.1007/s00158-022-03378-8
    """
    def __init__(self, solver: LinearSolver, tol=1e-8, A=None, hermitian=False, symmetric=False):
        self.solver = solver
        self.tol = tol
        self.x_stored = []
        self.b_stored = []
        self.xadj_stored = []
        self.badj_stored = []
        self.A = None
        self._did_solve = False  # For debugging purposes
        self._last_rtol = 0.
        self.hermitian = hermitian
        self.symmetric = symmetric
        super().__init__(A)

    def update(self, A):
        """ Clear the internal stored solution vectors and update the internal ``solver`` """
        self.A = A
        self.x_stored.clear()
        self.b_stored.clear()
        self.xadj_stored.clear()
        self.badj_stored.clear()
        self.solver.update(A)

    def _do_solve_1rhs(self, A, rhs, x_data, b_data, solve_fn):
        rhs_loc = rhs.copy()
        sol = 0

        # Check linear dependencies in the rhs using modified Gram-Schmidt
        for (x, b) in zip(x_data, b_data):
            alpha = rhs_loc.conj() @ b / (b.conj() @ b)
            rhs_loc -= alpha * b
            sol += alpha * x

        # Check tolerance
        self._last_rtol = 1.0 if len(x_data) == 0 else self.residual(A, sol, rhs)

        if self._last_rtol > self.tol:
            # Calculate a new solution
            xnew = solve_fn(rhs_loc)
            x_data.append(xnew)
            b_data.append(rhs_loc)
            sol += xnew
            self._did_solve = True
        else:
            self._did_solve = False

        return sol

    def _solve_1x(self, b):
        return self._do_solve_1rhs(self.A, b, self.x_stored, self.b_stored, self.solver.solve)

    def _adjoint_1x(self, b):
        return self._do_solve_1rhs(self.A.conj().T, b, self.xadj_stored, self.badj_stored, self.solver.adjoint)

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by performing a modified
        Gram-Schmidt over the previously calculated solutions :math:`\mathbf{U}` and corresponding right-hand-sides
        :math:`\mathbf{F}`. This is used to construct an approximate solution
        :math:`\tilde{\mathbf{x}} = \sum_k \alpha_k \mathbf{u}_k` in the subspace of :math:`\mathbf{U}`.
        If the residual of :math:`\mathbf{A} \tilde{\mathbf{x}} = \mathbf{b}` is above the tolerance, a new solution
        :math:`\mathbf{u}_{k+1}` will be added to the database such that
        :math:`\mathbf{x} = \tilde{\mathbf{x}}+\mathbf{u}_{k+1}` is the solution to the system
        :math:`\mathbf{A} \mathbf{x} = \mathbf{b}`.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        if rhs.ndim == 1:
            return self._solve_1x(rhs)
        else:  # Multiple rhs
            sol = []
            for i in range(rhs.shape[-1]):
                sol.append(self._solve_1x(rhs[..., i]))
            return np.stack(sol, axis=-1)

    def adjoint(self, rhs):
        if self.hermitian:
            return self.solve(rhs)
        elif self.symmetric:
            return self.solve(rhs.conj()).conj()
        else:
            if rhs.ndim == 1:
                return self._adjoint_1x(rhs)
            else:  # Multiple rhs
                sol = []
                for i in range(rhs.shape[-1]):
                    sol.append(self._adjoint_1x(rhs[..., i]))
                return np.stack(sol, axis=-1)
