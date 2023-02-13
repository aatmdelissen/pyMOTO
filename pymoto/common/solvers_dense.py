import warnings
import numpy as np
import scipy.linalg as spla  # Dense matrix solvers
from .solvers import matrix_is_hermitian, matrix_is_diagonal, LinearSolver


class SolverDiagonal(LinearSolver):
    """ Solver for diagonal matrices """
    def update(self, A):
        """ Extracts the diagonal of the matrix """
        self.diag = A.diagonal()
        return self

    def solve(self, rhs):
        r""" Solve using the diagonal only, by :math:`x_i = b_i / A_{ii}`

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        if rhs.ndim == 1:
            return rhs / self.diag
        else:
            return rhs / self.diag[..., None]

    def adjoint(self, rhs):
        r""" Solve using the diagonal only, by :math:`x_i = b_i / A_{ii}^*`

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        return self.solve(rhs.conj()).conj()


# Dense QR solver
class SolverDenseQR(LinearSolver):
    """ Solver for dense (square) matrices using a QR decomposition """
    def update(self, A):
        r""" Factorize the matrix as :math:`\mathbf{A}=\mathbf{Q}\mathbf{R}`, where :math:`\mathbf{Q}` is orthogonal
        (:math:`\mathbf{Q}^\text{H}=\mathbf{Q}^{-1}`) and :math:`\mathbf{R}` is upper triangular.
        """
        self.A = A
        self.q, self.r = spla.qr(A)
        return self

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by backward substitution of
        :math:`\mathbf{x} = \mathbf{R}^{-1}\mathbf{Q}^\text{H}\mathbf{b}`.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        return spla.solve_triangular(self.r, self.q.T.conj()@rhs)

    def adjoint(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A}^\text{H}\mathbf{x} = \mathbf{b}` by
        forward substitution of :math:`\mathbf{x} = \mathbf{Q}\mathbf{R}^{-H}\mathbf{b}`.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        return self.q@spla.solve_triangular(self.r, rhs, trans='C')


# Dense LU solver
class SolverDenseLU(LinearSolver):
    """ Solver for dense (square) matrices using an LU decomposition """
    def update(self, A):
        r"""  Factorize the matrix as :math:`\mathbf{A}=\mathbf{L}\mathbf{U}`, where :math:`\mathbf{L}` is a lower
        triangular matrix and :math:`\mathbf{U}` is upper triangular.
        """
        self.p, self.l, self.u = spla.lu(A)
        return self

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{U}^{-1}\mathbf{L}^{-1}\mathbf{b}`.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        return spla.solve_triangular(self.u, spla.solve_triangular(self.l, self.p.T@rhs, lower=True))

    def adjoint(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A}^\text{H}\mathbf{x} = \mathbf{b}` by forward and
        backward substitution of :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{U}^{-\text{H}}\mathbf{b}`.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        return self.p@spla.solve_triangular(self.l, spla.solve_triangular(self.u, rhs, trans='C'),
                                            lower=True, trans='C')  # TODO permutation


# Dense Cholesky solver
class SolverDenseCholesky(LinearSolver):
    """ Solver for Hermitian positive-definite matrices using a Cholesky factorization.
    In case the matrix is singular and factorization fails, a backup-solver is used (:class:`.SolverDenseLDL`).
    """
    def __init__(self, *args, **kwargs):
        self.backup_solver = SolverDenseLDL()
        self.success = None
        super().__init__(*args, **kwargs)

    def update(self, A):
        r""" Factorize the matrix as :math:`\mathbf{A}=\mathbf{U}^{\text{H}}\mathbf{U}`, where :math:`\mathbf{U}` is an
        upper triangular matrix.
        """
        try:
            self.u = spla.cholesky(A)
            self.success = True
        except np.linalg.LinAlgError as err:
            warnings.warn(f"{type(self).__name__}: {err} -- using {type(self.backup_solver).__name__} instead")
            self.backup_solver.update(A)
            self.success = False
        return self

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{U}^{-1}\mathbf{U}^{-\text{H}}\mathbf{b}`.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        # TODO When Cholesky factorization A = U^T U is used, symmetric complex matrices can also be solved, but this is
        #  not implemented in scipy
        if self.success:
            return spla.solve_triangular(self.u, spla.solve_triangular(self.u, rhs, trans='C'))
        else:
            return self.backup_solver.solve(rhs)

    def adjoint(self, rhs):
        r""" A Hermitian matrix is self-adjoint (:math:`\mathbf{A}=\mathbf{A}^\text{H}`), so this is equal to the
        regular solution.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        if self.success:
            return self.solve(rhs)
        else:
            return self.backup_solver.adjoint(rhs)


# Dense LDL solver
class SolverDenseLDL(LinearSolver):
    """ Solver for Hermitian or symmetric matrices using an LDL factorization. Unlike :class:`.SolverDenseCholesky`,
    it is able to factorize (some, not all) indefinite matrices, as well as symmetric complex (thus non-Hermitian)
    matrices.

    Requires scipy>=1.7
    """
    def __init__(self, *args, hermitian=None, **kwargs):
        self.hermitian = hermitian
        super().__init__(*args, **kwargs)

    def update(self, A):
        r""" Factorize the matrix as :math:`\mathbf{A}=\mathbf{L}\mathbf{D}\mathbf{L}^{\text{H}}` in case it is
        Hermitian, or as :math:`\mathbf{A}=\mathbf{L}\mathbf{D}\mathbf{L}^{\text{T}}` if it is symmetric. In the case
        matrix :math:`\mathbf{A}` is real-valued, there is no difference between the two.
        The matrix :math:`\mathbf{L}` is lower triangular and :math:`\mathbf{D}` is a diagonal matrix.
        """
        if self.hermitian is None:
            self.hermitian = matrix_is_hermitian(A)
        self.l, self.d, self.p = spla.ldl(A, hermitian=self.hermitian)  # LDL is introduced in Scipy v1.7
        if matrix_is_diagonal(self.d):  # Exact diagonal
            d1 = np.diag(1/np.diag(self.d))
        else:
            d1 = np.linalg.inv(self.d)  # This could be improved by looking at blocks on the diagonal
        self.dinv = lambda b: d1@b
        self.dinvH = lambda b: (d1.conj().T)@b
        self.lp = self.l[self.p, :]
        return self

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{D}^{-1}\mathbf{L}^{-1}\mathbf{b}` in the
        Hermitian case or as :math:`\mathbf{x} = \mathbf{L}^{-\text{T}}\mathbf{D}^{-1}\mathbf{L}^{-1}\mathbf{b}` in the
        symmetric case.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        u1 = spla.solve_triangular(self.lp, rhs[self.p], lower=True, unit_diagonal=True)
        u2 = self.dinv(u1)
        u = np.zeros_like(rhs, dtype=u2.dtype)
        u[self.p] = spla.solve_triangular(self.lp, u2, trans='C' if self.hermitian else 'T', lower=True, unit_diagonal=True)
        return u

    def adjoint(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A}^\text{H} \mathbf{x} = \mathbf{b}` by forward and
        backward substitution of
        :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{D}^{-\text{H}}\mathbf{L}^{-1}\mathbf{b}` in the  Hermitian
        case or as :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{D}^{-\text{H}}\mathbf{L}^{-*}\mathbf{b}`
        in the symmetric case.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        if not self.hermitian:
            u1 = spla.solve_triangular(self.lp, rhs[self.p].conj(), lower=True, unit_diagonal=True).conj()
        else:
            u1 = spla.solve_triangular(self.lp, rhs[self.p], lower=True, unit_diagonal=True)
        u2 = self.dinvH(u1)
        u = np.zeros_like(rhs, dtype=u2.dtype)
        u[self.p] = spla.solve_triangular(self.lp, u2, trans='C', lower=True, unit_diagonal=True)
        return u
