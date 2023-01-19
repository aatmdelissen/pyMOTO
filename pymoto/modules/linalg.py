""" Specialized linear algebra modules """
import os
import sys
import glob
import warnings
import hashlib
from inspect import currentframe, getframeinfo

from pymoto import Module, DyadCarrier, LDAWrapper
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spla  # Dense matrix solvers

from pymoto import SolverDenseLU, SolverDenseLDL, SolverDenseCholesky, SolverDiagonal, SolverDenseQR
from pymoto import matrix_is_symmetric, matrix_is_hermitian, matrix_is_diagonal
from pymoto import SolverSparseLU, SolverSparseCholeskyCVXOPT, SolverSparsePardiso, SolverSparseCholeskyScikit


class Inverse(Module):
    """ Calculate the exact inverse of a matrix B = A^{-1} """
    def _response(self, A):
        return np.linalg.inv(A)

    def _sensitivity(self, dB):
        A = self.sig_in[0].state
        B = self.sig_out[0].state
        dA = - B.T @ dB @ B.T
        return dA if np.iscomplexobj(A) else np.real(dA)


def auto_determine_solver(A, isdiagonal=None, islowertriangular=None, isuppertriangular=None, ishermitian=None, issymmetric=None, ispositivedefinite=None):
    """
    Uses parts of Matlab's scheme https://nl.mathworks.com/help/matlab/ref/mldivide.html
    :param A: The matrix
    :param isdiagonal: Manual override for diagonal matrix
    :param islowertriangular: Override for lower triangular matrix
    :param isuppertriangular: Override for upper triangular matrix
    :param ishermitian: Override for hermitian matrix (prevents check)
    :param issymmetric: Override for symmetric matrix (prevents check). Is the same as hermitian for a real matrix
    :param ispositivedefinite: Manual override for positive definiteness
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
        warnings.WarningMessage("Lower triangular solver not implemented", UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

    if isuppertriangular is None:  # Check if matrix is upper triangular
        isuppertriangular = False if issparse else np.allclose(A, np.triu(A))
    if isuppertriangular:
        warnings.WarningMessage("Upper triangular solver not implemented", UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

    ispermutedtriangular = False
    if ispermutedtriangular:
        warnings.WarningMessage("Permuted triangular solver not implemented", UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

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
        if SolverSparsePardiso.defined:
            # TODO check for positive definiteness?  np.alltrue(A.diagonal() > 0) or np.alltrue(A.diagonal() < 0)
            return SolverSparsePardiso(symmetric=issymmetric, hermitian=ishermitian, positive_definite=ispositivedefinite)

        if ishermitian:
            # Check if diagonal is all positive or all negative -> Cholesky
            if np.alltrue(A.diagonal() > 0) or np.alltrue(A.diagonal() < 0):  # TODO what about the complex case?
                if SolverSparseCholeskyScikit.defined:
                    return SolverSparseCholeskyScikit()
                if SolverSparseCholeskyCVXOPT.defined:
                    return SolverSparseCholeskyCVXOPT()

        return SolverSparseLU()  # Default to LU, which should be possible for any non-singular square matrix

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
        if not isinstance(self.solver, LDAWrapper):
            self.solver = LDAWrapper(self.solver)

        # Do factorication
        self.solver.update(mat)

        # Solution
        self.u = self.solver.solve(rhs)

        return self.u

    def _sensitivity(self, dfdv):
        mat, rhs = [s.state for s in self.sig_in]
        lam = self.solver.adjoint(dfdv)

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


class EigenSolve(Module):
    """ Eigensolver module
    Solves the eigenvalue problem A q_i = λ_i q_i with normalization q_i^T q_i = 1 or
    the generalized eigenvalue problem A q_i = λ_i B q_i with normalization q_i^T B q_i = 1 (B must be positive def.)
    The eigenvectors are returned as a matrix Q, where eigenvector Q[:, i] belongs to eigenvalue λ[i]

    Both real and complex matrices are supported.
    Correct sensitivities for cases with eigenvalue multiplicity are not implemented yet.

    Mode tracking algorithms can be implemented by the user by providing the argument "sorting_func", which is a
    function with arguments (λ, Q).
    """
    def _prepare(self, sorting_func=lambda W,Q: np.argsort(W), is_hermitian=None):

        self.sorting_fn = sorting_func
        self.is_hermitian = is_hermitian

    def _response(self, A, *args):
        B = args[0] if len(args) > 0 else None
        if self.is_hermitian is None:
            self.is_hermitian = (matrix_is_hermitian(A) and (B is None or matrix_is_hermitian(B)))
        W, Q = spla.eigh(A, b=B) if self.is_hermitian else spla.eig(A, b=B)
        isort = self.sorting_fn(W, Q)
        W = W[isort]
        Q = Q[:, isort]
        for i in range(W.size):
            qi, wi = Q[:, i], W[i]
            qi *= np.sign(np.real(qi[0]))
            Bqi = qi if B is None else B@qi
            qi /= np.sqrt(qi@Bqi)
            assert(abs(qi@(qi if B is None else B@qi) - 1.0) < 1e-5)
            assert(np.linalg.norm(A@qi - wi*(qi if B is None else B@qi)) < 1e-5)
        return W, Q

    def _sensitivity(self, dW, dQ):
        A = self.sig_in[0].state
        B = self.sig_in[1].state if len(self.sig_in) > 1 else np.eye(*A.shape)
        dA = np.zeros_like(A)
        dB = np.zeros_like(B) if len(self.sig_in) > 1 else None
        W, Q = [s.state for s in self.sig_out]
        if dW is None:
            dW = np.zeros_like(W)
        if dQ is None:
            dQ = np.zeros_like(Q)

        for i in range(W.size):
            dqi, dwi = dQ[:, i], dW[i]
            if np.linalg.norm(dqi) == 0 and dwi == 0:
                continue
            qi, wi = Q[:, i], W[i]

            P = np.block([[(A - wi*B).T, -((B + B.T)/2)@qi[..., np.newaxis]], [-B@qi.T, 0]])
            adj = np.linalg.solve(P, np.block([dqi, dwi]))  # Adjoint solve Lee(1999)
            nu = adj[:-1]
            alpha = adj[-1]
            dAi = np.outer(-nu, qi)
            dA += dAi if np.iscomplexobj(A) else np.real(dAi)
            if dB is not None:
                dBi = np.outer(wi*nu + alpha/2*qi, qi)
                dB += dBi if np.iscomplexobj(B) else np.real(dBi)

        if len(self.sig_in) == 1:
            return dA
        elif len(self.sig_in) == 2:
            return dA, dB
