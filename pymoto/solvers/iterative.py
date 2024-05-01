import warnings
import numpy as np
from .solvers import LinearSolver


class Preconditioner(LinearSolver):
    def update(self, A):
        pass

    def solve(self, rhs):
        return rhs.copy()

    def adjoint(self, rhs):
        return self.solve(rhs.conj()).conj()


class Jacobi(Preconditioner):
    def update(self, A):
        self.diag = A.diagonal()

    def solve(self, rhs):
        return (rhs.T/self.diag).T


def orth(u, normalize=True, zero_rtol=1e-15):
    """ Create orthogonal basis from a set of vectors

    Args:
        u: Set of vectors of size (#dof, #vectors)
        normalize: Also normalize the basis vectors
        zero_rtol: Relative tolerance for detection of zero vectors (in case of a rank-deficient basis)

    Returns:
        v: Orthogonal basis vectors (#dof, #non-zero-vectors)
    """
    if u.ndim == 1:
        return u
    elif u.ndim > 2:
        raise TypeError("Only valid for 1D or 2D matrix")

    def dot(a, b):  # Define inner product
        return a @ b.conj()

    orth_vecs = []
    for i in range(u.shape[-1]):
        vi = np.copy(u[..., i])
        beta_i = dot(vi, vi)
        for vj in orth_vecs:
            alpha_ij = dot(vi, vj)
            alpha_jj = 1.0 if normalize else dot(vj, vj)
            vi -= vj * alpha_ij / alpha_jj
        beta_i_new = dot(vi, vi)
        if beta_i_new / beta_i < zero_rtol:  # Detect zero vector
            continue
        if normalize:
            vi /= np.sqrt(beta_i_new)
        orth_vecs.append(vi)
    return np.stack(orth_vecs, axis=-1)


class CG(LinearSolver):
    """ Preconditioned conjugate gradient method

    References:
        Ji & Li (2017), A breakdown-free BCG method. DOI 10.1007/s10543-016-0631-z
          https://www.cs.odu.edu/~yaohang/portfolio/BIT2017.pdf
        Shewchuck (1994), Introduction to CG method without the agonzing pain.
          https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

    Args:
        A: The matrix
        preconditioner: Preconditioner to use
        tol: Convergence tolerance
        maxit: Maximum number of iterations
        restart: Restart every Nth iteration
        verbosity: Log level
    """
    def __init__(self, A=None, preconditioner=Preconditioner(), tol=1e-5, maxit=10000, restart=50, verbosity=0):
        self.preconditioner = preconditioner
        self.A = A
        self.tol = tol
        self.maxit = maxit
        self.restart = restart
        self.verbosity = verbosity
        super().__init__(A)

    def update(self, A):
        self.A = A
        self.preconditioner.update(A)

    def solve(self, rhs, x0=None):
        if rhs.ndim == 1:
            b = rhs.reshape((rhs.size, 1))
        else:
            b = rhs
        x = np.zeros_like(rhs, dtype=np.result_type(rhs, self.A)) if x0 is None else x0.copy()
        if x.ndim == 1:
            x = x.reshape((x.size, 1))

        r = b - self.A@x
        z = self.preconditioner.solve(r)
        p = orth(z, normalize=True)
        if self.verbosity >= 2:
            print(f"Initial residual = {np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=0)}")
        tol = 1e-5
        for i in range(self.maxit):
            q = self.A @ p
            pq = p.T @ q
            pq_inv = np.linalg.inv(pq)
            alpha = pq_inv @ (p.T @ r)

            x += p @ alpha
            if i % 50 == 0:  # Explicit restart
                r = b - self.A@x
            else:
                r -= q @ alpha

            if self.verbosity >= 2:
                print(f"i = {i}, residuals = {np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=0)}")

            tval = np.linalg.norm(r)/np.linalg.norm(b)
            if tval <= tol:
                break

            z = self.preconditioner.solve(r)

            beta = -pq_inv @ (q.T @ z)
            p = orth(z + p@beta, normalize=False)

        if tval > tol:
            warnings.warn(f'Maximum iterations ({self.maxit}) reached, with final residual {tval}')
        elif self.verbosity >= 1:
            print(f"Converged in {i} iterations, with final residuals {np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=0)}")

        if rhs.ndim == 1:
            return x.flatten()
        else:
            return x

    def adjoint(self, rhs):
        return self.solve(rhs.conj()).conj()