import warnings
import numpy as np
import scipy.linalg as spla  # Dense matrix solvers
from .solvers import matrix_is_hermitian

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
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    @staticmethod
    def residual(A, x, b):
        """ Calculates the residual || A x - b || / || b ||
        :param A: Matrix
        :param x: Solution
        :param b: Right-hand side
        :return: Residual value
        """
        return np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)

class SolverDiagonal(LinearSolver):
    def update(self, A):
        self.diag = A.diagonal()
        return self

    def solve(self, rhs):
        return rhs/self.diag

    def adjoint(self, rhs):
        return rhs/(self.diag.conj())


# Dense QR solver
class SolverDenseQR(LinearSolver):
    def update(self, A):
        self.A = A
        self.q, self.r = spla.qr(A)
        return self

    def solve(self, rhs):
        return spla.solve_triangular(self.r, self.q.T.conj()@rhs)

    def adjoint(self, rhs):
        return self.q@spla.solve_triangular(self.r, rhs, trans='C')


# Dense LU solver
class SolverDenseLU(LinearSolver):
    def update(self, A):
        self.p, self.l, self.u = spla.lu(A)
        return self

    def solve(self, rhs):
        return spla.solve_triangular(self.u, spla.solve_triangular(self.l, self.p.T@rhs, lower=True))

    def adjoint(self, rhs):
        return self.p@spla.solve_triangular(self.l, spla.solve_triangular(self.u, rhs, trans='C'), lower=True, trans='C')  # TODO permutation


# Dense Cholesky solver
class SolverDenseCholesky(LinearSolver):
    """ Only for Hermitian positive-definite matrix """
    def __init__(self, *args, **kwargs):
        self.backup_solver = SolverDenseLDL()
        self.success = None
        super().__init__(*args, **kwargs)

    def update(self, A):
        try:
            self.u = spla.cholesky(A)
            self.success = True
        except np.linalg.LinAlgError as err:
            warnings.warn(f"{type(self).__name__}: {err} -- using {type(self.backup_solver).__name__} instead")
            self.backup_solver.update(A)
            self.success = False
        return self

    def solve(self, rhs):
        if self.success:
            return spla.solve_triangular(self.u, spla.solve_triangular(self.u, rhs, trans='C'))
        else:
            return self.backup_solver.solve(rhs)

    def adjoint(self, rhs):
        if self.success:
            return self.solve(rhs)
        else:
            return self.backup_solver.adjoint(rhs)


# Dense LDL solver
class SolverDenseLDL(LinearSolver):
    def __init__(self, *args, hermitian=None, **kwargs):
        self.hermitian = hermitian
        super().__init__(*args, **kwargs)

    def update(self, A):
        if self.hermitian is None:
            self.hermitian = matrix_is_hermitian(A)
        self.l, self.d, self.p = spla.ldl(A, hermitian=self.hermitian)
        self.diagonald = np.allclose(self.d, np.diag(np.diag(self.d)))
        if self.diagonald:  # Exact diagonal
            d1 = np.diag(1/np.diag(self.d))
        else:
            d1 = np.linalg.inv(self.d)  # TODO, this could be improved
        self.dinv = lambda b: d1@b
        self.dinvH = lambda b: (d1.conj().T)@b
        self.lp = self.l[self.p, :]
        return self

    def solve(self, rhs):
        u1 = spla.solve_triangular(self.lp, rhs[self.p], lower=True, unit_diagonal=True)
        u2 = self.dinv(u1)
        u = np.zeros_like(rhs)
        u[self.p] = spla.solve_triangular(self.lp, u2, trans='C' if self.hermitian else 'T', lower=True, unit_diagonal=True)
        return u

    def adjoint(self, rhs):
        if not self.hermitian:
            u1 = spla.solve_triangular(self.lp, rhs[self.p].conj(), lower=True, unit_diagonal=True).conj()
        else:
            u1 = spla.solve_triangular(self.lp, rhs[self.p], lower=True, unit_diagonal=True)
        u2 = self.dinvH(u1)
        u = np.zeros_like(rhs)
        u[self.p] = spla.solve_triangular(self.lp, u2, trans='C', lower=True, unit_diagonal=True)
        return u