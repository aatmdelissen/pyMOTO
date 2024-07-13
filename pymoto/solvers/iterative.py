import warnings
import time
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu, spilu
from .solvers import LinearSolver
from .auto_determine import auto_determine_solver
from pymoto import DomainDefinition


class Preconditioner(LinearSolver):
    """ Abstract base class for preconditioners to inexact solvers """
    def update(self, A):
        pass

    def solve(self, rhs, x0=None, trans='N'):
        return rhs.copy()


class DampedJacobi(Preconditioner):
    r""" Damped Jacobi preconditioner
    :math:`M = \frac{1}{\omega} D`
    Args:
        A (optional): The matrix
        w (optional): Weight factor :math:`0 < \omega \leq 1`
    """
    def __init__(self, A=None, w=1.0):
        assert 0 < w <= 1, 'w must be between 0 and 1'
        self.w = w
        self.D = None
        super().__init__(A)

    def update(self, A):
        self.D = A.diagonal()

    def solve(self, rhs, x0=None, trans='N'):
        if trans == 'N' or trans == 'T':
            return self.w * (rhs.T/self.D).T
        elif trans == 'H':
            return self.w * (rhs.T/self.D.conj()).T
        else:
            raise TypeError("Only N, T, or H transposition is possible")


class SOR(Preconditioner):
    r""" Successive over-relaxation preconditioner
    The matrix :math:`A = L + D + U` is split into a lower triangular, diagonal, and upper triangular part.
    :math:`M = \left(\frac{D}{\omega} + L\right) \frac{\omega D^{-1}}{2-\omega} \left(\frac{D}{\omega} + U\right)`

    Args:
        A (optional): The matrix
        w (optional): Weight factor :math:`0 < \omega < 2`
    """
    def __init__(self, A=None, w=1.0):
        assert 0 < w < 2, 'w must be between 0 and 2'
        self.w = w
        self.L = None
        self.U = None
        self.Dw = None
        super().__init__(A)

    def update(self, A):
        diag = A.diagonal()
        diagw = sps.diags(diag)/self.w
        self.L = splu(sps.tril(A, k=-1) + diagw)  # Lower triangular part including diagonal
        self.U = splu(sps.triu(A, k=1) + diagw)

        self.Dw = diag * (2 - self.w) / self.w

    def solve(self, rhs, x0=None, trans='N'):
        if trans == 'N':
            # M = (D/w + L) wD^-1 / (2-w) (D/w + U)
            # from scipy.sparse.linalg import spsolve_triangular
            # u1 = spsolve_triangular(self.L, rhs, lower=True, overwrite_A=False)  # Solve triangular is still very slow :(
            u1 = self.L.solve(rhs)
            u1 *= self.Dw[:, None]
            # u2 = spsolve_triangular(self.U, u1, lower=False, overwrite_A=False, overwrite_b=True)
            u2 = self.U.solve(u1)
            return u2
        elif trans == 'T':
            u1 = self.U.solve(rhs, trans='T')
            u1 *= self.Dw[:, None]
            u2 = self.L.solve(u1, trans='T')
            return u2
        elif trans == 'H':
            u1 = self.U.solve(rhs, trans='H')
            u1 *= self.Dw[:, None].conj()
            u2 = self.L.solve(u1, trans='H')
            return u2
        else:
            raise TypeError("Only N, T, or H transposition is possible")


class ILU(Preconditioner):
    """ Incomplete LU factorization

    Args:
        A (optional): The matrix
        **kwargs (optional): Keyword arguments passed to `scipy.sparse.linalg.spilu`
    """
    def __init__(self, A=None, **kwargs):
        self.kwargs = kwargs
        self.ilu = None
        super().__init__(A)

    def update(self, A):
        self.ilu = spilu(A, **self.kwargs)

    def solve(self, rhs, x0=None, trans='N'):
        return self.ilu.solve(rhs, trans=trans)


class GeometricMultigrid(Preconditioner):
    """ Geometric multigrid preconditioner

    Args:
        domain: The `DomainDefinition` used for the geometry
        A (optional): The matrix
        inner_level (optional): Inner solver for the coarse grid, for instance, a direct solver or another MG level.
            The default is an automatically determined direct solver.
        smoother (optional): Smoother to use to smooth the residual and solution before and after coarse level.
            The default is `DampedJacobi(w=0.5)`.
        smooth_steps (optional): Number of smoothing steps to execute
    """
    _available_cycles = ['v', 'w']

    def __init__(self, domain: DomainDefinition, A=None, cycle='V', inner_level=None, smoother=None, smooth_steps=5):
        assert domain.nelx % 2 == 0 and domain.nely % 2 == 0 and domain.nelz % 2 == 0, \
            f"Domain sizes {domain.nelx, domain.nely, domain.nelz} must be divisible by 2"
        self.domain = domain
        self.A = A
        assert cycle.lower() in self._available_cycles, f"Cycle ({cycle}) is not available. Options are {self._available_cycles}"
        self.cycle = cycle
        self.inner_level = None if inner_level is None else inner_level
        self.smoother = DampedJacobi(w=0.5) if smoother is None else None
        self.smooth_steps = smooth_steps
        self.R = None
        self.sub_domain = DomainDefinition(domain.nelx // 2, domain.nely // 2, domain.nelz // 2,
                                           domain.unitx * 2, domain.unity * 2, domain.unitz * 2)

        super().__init__(A)

    def update(self, A):
        if self.R is None:
            self.setup_interpolation(A)
        self.A = A
        self.smoother.update(A)
        Ac = self.R.T @ A @ self.R
        if self.inner_level is None:
            self.inner_level = auto_determine_solver(Ac)
        self.inner_level.update(Ac)

    def setup_interpolation(self, A):
        assert A.shape[0] % self.domain.nnodes == 0
        ndof = int(A.shape[0] / self.domain.nnodes)  # Number of dofs per node

        w = np.ones((3, 3, 3))*0.125
        w[1, :, :] = 0.25
        w[:, 1, :] = 0.25
        w[:, :, 1] = 0.25
        w[1, 1, :] = 0.5
        w[1, :, 1] = 0.5
        w[:, 1, 1] = 0.5
        w[1, 1, 1] = 1.0

        rows = []
        cols = []
        vals = []
        for i in [-1, 0, 1]:
            imin, imax = max(-i, 0), min(self.sub_domain.nelx + 1 - i, self.sub_domain.nelx + 1)
            ix = np.arange(imin, imax)
            for j in [-1, 0, 1]:
                jmin, jmax = max(-j, 0), min(self.sub_domain.nely + 1 - j, self.sub_domain.nely + 1)
                iy = np.arange(jmin, jmax)
                for k in ([-1, 0, 1] if self.domain.dim == 3 else [0]):
                    # Coarse node cartesian indices
                    kmin, kmax = max(-k, 0), min(self.sub_domain.nelz + 1 - k, self.sub_domain.nelz + 1)
                    iz = np.arange(kmin, kmax)
                    # Coarse node numbers
                    nod_c = self.sub_domain.get_nodenumber(*np.meshgrid(ix, iy, iz, indexing='ij')).flatten()
                    # Fine node numbers with offset
                    ixc, iyc, izc = ix * 2 + i, iy * 2 + j, iz * 2 + k
                    nod_f = self.domain.get_nodenumber(*np.meshgrid(ixc, iyc, izc, indexing='ij')).flatten()
                    for d in range(ndof):
                        rows.append(nod_f * ndof + d)
                        cols.append(nod_c * ndof + d)
                        vals.append(np.ones_like(rows[-1], dtype=w.dtype) * w[1+i, 1+j, 1+k])

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        nfine = ndof * self.domain.nnodes
        ncoarse = ndof * self.sub_domain.nnodes
        self.R = sps.coo_matrix((vals, (rows, cols)), shape=(nfine, ncoarse))
        self.R = type(A)(self.R)  # Convert to correct matrix type

    def solve(self, rhs, x0=None, trans='N'):
        if trans == 'N':
            A = self.A
        elif trans == 'T':
            A = self.A.T
        elif trans == 'H':
            A = self.A.conj().T
        else:
            raise TypeError("Only N, T, or H transposition is possible")

        # Pre-smoothing
        if x0 is None:
            u_f = self.smoother.solve(rhs, trans=trans)
        else:
            r = rhs - self.A @ x0
            u_f = x0 + self.smoother.solve(r, trans=trans)
        for i in range(self.smooth_steps-1):
            r = rhs - self.A @ u_f
            u_f += self.smoother.solve(r, trans=trans)

        r = rhs - A @ u_f
        # Restrict residual to coarse level
        r_c = self.R.T @ r

        # Solve at coarse level
        u_c = self.inner_level.solve(r_c)

        # Interpolate and correct
        u_f += self.R @ u_c

        # Post-smoothing
        for i in range(self.smooth_steps):
            r = rhs - self.A @ u_f
            u_f += self.smoother.solve(r, trans=trans)
        return u_f


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
    Works for positive-definite self-adjoint matrices (:math:`A=A^H`)

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
    def __init__(self, A=None, preconditioner=Preconditioner(), tol=1e-7, maxit=10000, restart=50, verbosity=0):
        self.preconditioner = preconditioner
        self.A = A
        self.tol = tol
        self.maxit = maxit
        self.restart = restart
        self.verbosity = verbosity
        super().__init__(A)

    def update(self, A):
        tstart = time.perf_counter()
        self.A = A
        self.preconditioner.update(A)
        if self.verbosity >= 1:
            print(f"Preconditioner set up in {np.round(time.perf_counter() - tstart,3)}s")

    def solve(self, rhs, x0=None, trans='N'):
        if trans == 'N':
            A = self.A
        elif trans == 'T':
            A = self.A.T
        elif trans == 'H':
            A = self.A.conj().T
        else:
            raise TypeError("Only N, T, or H transposition is possible")

        tstart = time.perf_counter()
        if rhs.ndim == 1:
            b = rhs.reshape((rhs.size, 1))
        else:
            b = rhs
        x = np.zeros_like(rhs, dtype=np.result_type(rhs, A)) if x0 is None else x0.copy()
        if x.ndim == 1:
            x = x.reshape((x.size, 1))

        r = b - A@x
        z = self.preconditioner.solve(r, trans=trans)
        p = orth(z, normalize=True)
        if self.verbosity >= 2:
            print(f"Initial residual = {np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=0)}")

        for i in range(self.maxit):
            q = A @ p
            pq = p.conj().T @ q
            pq_inv = np.linalg.inv(pq)
            alpha = pq_inv @ (p.conj().T @ r)

            x += p @ alpha
            if i % 50 == 0:  # Explicit restart
                r = b - A@x
            else:
                r -= q @ alpha

            if self.verbosity >= 2:
                print(f"i = {i}, residuals = {np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=0)}")

            tval = np.linalg.norm(r)/np.linalg.norm(b)
            if tval <= self.tol:
                break

            z = self.preconditioner.solve(r, trans=trans)

            beta = -pq_inv @ (q.conj().T @ z)
            p = orth(z + p@beta, normalize=False)

        if tval > self.tol:
            warnings.warn(f'Maximum iterations ({self.maxit}) reached, with final residual {tval}')
        elif self.verbosity >= 1:
            print(f"Converged in {i} iterations and {np.round(time.perf_counter() - tstart, 3)}s, with final residuals {np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=0)}")

        if rhs.ndim == 1:
            return x.flatten()
        else:
            return x
