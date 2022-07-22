""" Specialized linear algebra modules """
import os
import sys
import glob
import warnings
import hashlib

from pymodular import Module, DyadCarrier
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spla  # Dense matrix solvers

from pymodular import SolverDenseLU, SolverDenseLDL, SolverDenseCholesky, SolverDiagonal, SolverDenseQR
from pymodular import matrix_is_symmetric, matrix_is_hermitian, matrix_is_diagonal
from pymodular import SolverSparseLU, SolverSparseCholeskyCVXOPT, SolverSparsePardiso, SolverSparseCholeskyScikit


def auto_determine_solver(A, isdiagonal=None, islowertriangular=None, isuppertriangular=None, ishermitian=None, issymmetric=None):
    """
    Uses parts of Matlab's scheme https://nl.mathworks.com/help/matlab/ref/mldivide.html
    :param A: The matrix
    :return: LinearSolver which should be 'best' for the matrix
    """
    issparse = sps.issparse(A)  # Check if the matrix is sparse
    issquare = A.shape[0] == A.shape[1]  # Check if the matrix is square

    if not issquare:
        if issparse:
            sps.SparseEfficiencyWarning("Only a dense version of QR solver is available")  # TODO
        return SolverDenseQR()

    # l_bw, u_bw = spla.bandwidth(A) # TODO Get bandwidth (implemented in scipy version > 1.8.0)

    if isdiagonal is None:  # Check if matrix is diagonal
        # TODO: This could be improved to check other sparse matrix types as well
        isdiagonal = matrix_is_diagonal(A)
    if isdiagonal:
        return SolverDiagonal()

    # Check if the matrix is triangular
    # TODO Currently only for dense matrices
    if islowertriangular is None:  # Check if matrix is lower triangular
        islowertriangular = False if issparse else np.allclose(A, np.tril(A))
    if islowertriangular:
        warnings.WarningMessage("Lower triangular solver not implemented")

    if isuppertriangular is None:  # Check if matrix is upper triangular
        isuppertriangular = False if issparse else np.allclose(A, np.triu(A))
    if isuppertriangular:
        warnings.WarningMessage("Upper triangular solver not implemented")

    ispermutedtriangular = False
    if ispermutedtriangular:
        warnings.WarningMessage("Permuted triangular solver not implemented")

    # Check if the matrix is complex-valued
    iscomplex = np.iscomplexobj(A)
    if iscomplex:
        # Detect if the matrix is hermitian and/or symmetric
        if ishermitian is None:
            ishermitian = matrix_is_hermitian(A)
        if issymmetric is None:
            issymmetric = matrix_is_symmetric(A)
    else:
        if ishermitian is None and issymmetric is None:
            # Detect if the matrix is symmetric
            issymmetric = matrix_is_symmetric(A)
            ishermitian = issymmetric
        elif ishermitian is not None and issymmetric is not None:
            assert ishermitian == issymmetric, "For real-valued matrices, symmetry and hermitian must be equal"
        elif ishermitian is None:
            ishermitian = issymmetric
        elif issymmetric is None:
            issymmetric = ishermitian

    if issparse:
        # Prefer Intel Pardiso solver as it can solve any matrix TODO: Check for complex matrix
        """
        mtype:     1  real and structurally symmetric
                   2  real and symmetric positive definite
                  -2  real and symmetric indefinite
                   3  complex and structurally symmetric
                   4  complex and Hermitian positive definite
                  -4  complex and Hermitian indefinite
                   6  complex and symmetric
                   11 real and nonsymmetric
                   13 complex and nonsymmetric
        """
        if SolverSparsePardiso.defined:
            if iscomplex:
                if ishermitian:
                    mtype = -4  # TODO check for positive definiteness?
                elif issymmetric:
                    mtype = 6
                else:
                    mtype = 13
            else:  # real
                if issymmetric:
                    mtype = 2 if np.alltrue(A.diagonal() > 0) or np.alltrue(A.diagonal() < 0) else -2
                else:
                    mtype = 11
            return SolverSparsePardiso(mtype=mtype)

        if ishermitian:
            # Check if diagonal is all positive or all negative -> Cholesky
            if np.alltrue(A.diagonal() > 0) or np.alltrue(A.diagonal() < 0):
                if SolverSparseCholeskyScikit.defined:
                    return SolverSparseCholeskyScikit()
                if SolverSparseCholeskyCVXOPT.defined:
                    return SolverSparseCholeskyCVXOPT()

        return SolverSparseLU()

    else:  # Dense
        if ishermitian:
            # Check if diagonal is all positive or all negative
            if np.alltrue(A.diagonal() > 0) or np.alltrue(A.diagonal() < 0):
                return SolverDenseCholesky()
            else:
                return SolverDenseLDL(hermitian=ishermitian)
        elif issymmetric:
            return SolverDenseLDL(hermitian=ishermitian)
        else:
            # TODO: Detect if the matrix is Hessenberg
            return SolverDenseLU()


class LinSolve(Module):
    """ Linear solver module
    Solves linear system of equations Ax=b
    """
    def _prepare(self, dep_tol=1e-5, hermitian=None, symmetric=None, solver=None):
        """
        :param tol: Tolerance for detecting linear dependence of adjoint vector
        :param hermitian: Flag to omit the detection for hermitian matrix, saves some work for large matrices
        :param solver: Provide a custom LinearSolver to use that instead of the 'automatic' solver
        """
        self.dep_tol = dep_tol
        self.ishermitian = hermitian
        self.issymmetric = symmetric
        self.solver = solver

    def _response(self, mat, rhs):
        # Do some detections on the matrix type
        self.issparse = sps.issparse(mat)  # Check if it is a sparse matrix
        self.iscomplex = np.iscomplexobj(mat)  # Check if it is a complex-valued matrix
        if not self.iscomplex and self.issymmetric is not None:
            self.ishermitian = self.issymmetric
        if self.ishermitian is None:
            self.ishermitian = matrix_is_hermitian(mat)

        # Determine the solver we want to use
        if self.solver is None:
            self.solver = auto_determine_solver(mat, ishermitian=self.ishermitian)

        # Do factorication
        self.solver.update(mat)

        # Solution
        self.u = self.solver.solve(rhs)

        # assert np.allclose(mat@self.u, rhs)
        return self.u

    def _sensitivity(self, dfdv):
        dfdu = dfdv
        rhs = self.sig_in[1].state

        # Check if b is linear dependent on dfdu
        # Asymmetric matrices are not self-adjoint
        islinear = self.ishermitian

        # Check if the adjoint rhs vector is linearly dependent on rhs vector b
        if islinear:
            # Projection of dfdu onto right-hand-side
            if dfdu.ndim > 1:
                islinear = False  # Just do the linear solve for now
                # TODO: Multiple rhs might be solved by linear combination of previous solutions
            else:
                dfdunorm = dfdu.dot(dfdu)
                alpha = rhs.dot(dfdu) / dfdunorm
                islinear = np.linalg.norm(dfdu - alpha * rhs) / dfdunorm < self.dep_tol

        if islinear:
            lam = alpha * self.u
        else:
            lam = self.solver.adjoint(dfdu)

        if self.issparse:
            if not self.iscomplex and (np.iscomplexobj(self.u) or np.iscomplexobj(lam)):
                warnings.warn("This one has not been checked yet!")  # TODO
                dmat = DyadCarrier([-np.real(lam), -np.imag(lam)], [np.real(self.u), np.imag(self.u)])
            else:
                dmat = DyadCarrier(-lam, self.u)
        else:
            if self.u.ndim > 1:
                dmat = np.einsum("iB,jB->ij", -lam, np.conj(self.u))
            else:
                dmat = np.outer(-lam, np.conj(self.u))
            if not self.iscomplex:
                dmat = np.real(dmat)

        db = np.real(lam) if np.isrealobj(rhs) else lam

        return dmat, db
