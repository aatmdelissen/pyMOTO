""" Specialized linear algebra modules """
from pyModular.core_objects import Module
from pyModular.dyadcarrier import DyadCarrier
import numpy as np
import scipy.sparse as sps


class LinSolve(Module):
    """ Linear solver module
    Solves linear system of equations Ax=b
    """
    def _prepare(self, tol=1e-5, symmetric=None):
        """
        :param tol: Tolerance for detecting linear dependence of adjoint vector
        :param symmetric: Flag to omit the detection for symmetric matrix, saves some work for large matrices
        :return:
        """
        self.tol = tol
        if symmetric is not None:
            self.issymmetric = symmetric

    def _response(self, mat, rhs):
        self.issparse = sps.issparse(mat)
        self.iscomplex = np.iscomplexobj(mat)

        # Do an LU factorization
        if self.issparse:  # For sparse matrix
            import scipy.sparse.linalg as spsla
            self.inv = spsla.splu(mat)
            self.u = self.inv.solve(rhs)
            self.adjointsolve = lambda b: self.inv.solve(b, trans=('H' if self.iscomplex else 'T'))

        else:  # For dense matrix
            import scipy.linalg as spla
            self.inv = spla.lu_factor(mat)
            self.u = spla.lu_solve(self.inv, rhs)
            self.adjointsolve = lambda b: spla.lu_solve(self.inv, b, trans=(2 if self.iscomplex else 1))

        # Detect matrix symmetry on first run
        if not hasattr(self, 'issymmetric'):
            if self.issparse:
                self.issymmetric = (abs(mat-mat.T) > 1e-10).nnz == 0
            else:
                self.issymmetric = np.allclose(mat, mat.T)

        return self.u

    def _sensitivity(self, dfdv):
        dfdu = np.conj(dfdv)
        rhs = self.sig_in[1].state

        # Check if b is linear dependent on dfdu
        # Asymmetric matrices are not self-adjoint
        islinear = self.issymmetric

        # Check if the adjoint rhs vector is linearly dependent on rhs vector b
        if islinear:
            # Projection of dfdu onto right-hand-side
            # TODO: Check for multiple rhs
            dfdunorm = dfdu.dot(dfdu)
            alpha = rhs.dot(dfdu) / dfdunorm
            tol = 1e-5
            islinear = np.linalg.norm(dfdu - alpha * rhs) / dfdunorm < tol

        if islinear:
            lam = alpha * self.u
        else:
            lam = self.adjointsolve(dfdu)

        if self.issparse:
            if not self.iscomplex and np.iscomplexobj(self.u):
                dmat = DyadCarrier([-np.real(lam), np.imag(lam)], [np.real(self.u), np.imag(self.u)])
            else:
                dmat = DyadCarrier(-lam, self.u)
        else:
            dmat = np.outer(-np.conj(lam), np.conj(self.u))
            if self.iscomplex:
                dmat = np.real(dmat)

        db = np.real(lam) if np.isrealobj(rhs) else np.conj(lam)

        return dmat, db
