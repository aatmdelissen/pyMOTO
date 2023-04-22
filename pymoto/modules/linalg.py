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
import scipy.sparse.linalg as spsla
import scipy.linalg as spla  # Dense matrix solvers

from pymoto import SolverDenseLU, SolverDenseLDL, SolverDenseCholesky, SolverDiagonal, SolverDenseQR
from pymoto import matrix_is_symmetric, matrix_is_hermitian, matrix_is_diagonal
from pymoto import SolverSparseLU, SolverSparseCholeskyCVXOPT, SolverSparsePardiso, SolverSparseCholeskyScikit


class Inverse(Module):
    r""" Calculate the inverse of a matrix :math:`\mathbf{B} = \mathbf{A}^{-1}`

    Input Signal:
      - ``A`` (`np.ndarray`): Dense matrix:math:`\mathbf{A}`

    Output Signal:
      - `B` (`np.ndarray`): The inverse of :math:`\mathbf{A}` as dense matrix
    """
    def _response(self, A):
        return np.linalg.inv(A)

    def _sensitivity(self, dB):
        A = self.sig_in[0].state
        B = self.sig_out[0].state
        dA = - B.T @ dB @ B.T
        return dA if np.iscomplexobj(A) else np.real(dA)


# flake8: noqa: C901
def auto_determine_solver(A, isdiagonal=None, islowertriangular=None, isuppertriangular=None,
                          ishermitian=None, issymmetric=None, ispositivedefinite=None):
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
        warnings.WarningMessage("Lower triangular solver not implemented",
                                UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

    if isuppertriangular is None:  # Check if matrix is upper triangular
        isuppertriangular = False if issparse else np.allclose(A, np.triu(A))
    if isuppertriangular:
        warnings.WarningMessage("Upper triangular solver not implemented",
                                UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

    ispermutedtriangular = False
    if ispermutedtriangular:
        warnings.WarningMessage("Permuted triangular solver not implemented",
                                UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

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
        if SolverSparsePardiso.defined and not iscomplex:
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

    else:  # Dense branch
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
    r""" Solves linear system of equations :math:`\mathbf{A}\mathbf{x}=\mathbf{b}`

    Self-adjointness is automatically detected using :class:`LDAWrapper`.

    Input Signals:
      - ``A`` (`dense or sparse matrix`): The system matrix :math:`\mathbf{A}` of size ``(n, n)``
      - ``b`` (`vector`): Right-hand-side vector of size ``(n)`` or block-vector of size ``(Nrhs, n)``

    Output Signal:
      - ``x`` (`vector`): Solution vector of size ``(n)`` or block-vector of size ``(Nrhs, n)``

    Keyword Args:
        dep_tol: Tolerance for detecting linear dependence of solution vectors (default = ``1e-5``)
        hermitian: Flag to omit the automatic detection for Hermitian matrix, saves some work for large matrices
        symmetric: Flag to omit the automatic detection for symmetric matrix, saves some work for large matrices
        solver: Manually override the LinearSolver used, instead of the the solver from :func:`auto_determine_solver`

    Attributes:
        use_lda_solver: Use the linear-dependency-aware solver :class:`LDAWrapper` to prevent redundant computations
    """
    use_lda_solver = True

    def _prepare(self, dep_tol=1e-5, hermitian=None, symmetric=None, solver=None):
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
        if self.issparse and not self.iscomplex and np.iscomplexobj(rhs):
            raise TypeError("Complex right-hand-side for a real-valued sparse matrix is not supported")

        # Determine the solver we want to use
        if self.solver is None:
            self.solver = auto_determine_solver(mat, ishermitian=self.ishermitian)
        if not isinstance(self.solver, LDAWrapper) and self.use_lda_solver:
            self.solver = LDAWrapper(self.solver, hermitian=self.ishermitian, symmetric=self.issymmetric)

        # Update solver with new matrix
        self.solver.update(mat)

        # Solution
        self.u = self.solver.solve(rhs)

        return self.u

    def _sensitivity(self, dfdv):
        mat, rhs = [s.state for s in self.sig_in]
        lam = self.solver.adjoint(dfdv.conj()).conj()

        if self.issparse:
            if not self.iscomplex and (np.iscomplexobj(self.u) or np.iscomplexobj(lam)):
                raise TypeError("Complex right-hand-side for a real-valued sparse matrix is not supported")  # TODO
                dmat = DyadCarrier([-np.real(lam), -np.imag(lam)], [np.real(self.u), np.imag(self.u)])
            else:
                dmat = DyadCarrier(-lam, self.u)
        else:
            if self.u.ndim > 1:
                dmat = np.einsum("iB,jB->ij", -lam, self.u, optimize=True)
            else:
                dmat = np.outer(-lam, self.u)
            if not self.iscomplex:
                dmat = np.real(dmat)

        db = np.real(lam) if np.isrealobj(rhs) else lam

        return dmat, db


class EigenSolve(Module):
    r""" Solves the (generalized) eigenvalue problem :math:`\mathbf{A}\mathbf{q}_i = \lambda_i \mathbf{B} \mathbf{q}_i`

    Solves the eigenvalue problem :math:`\mathbf{A}\mathbf{q}_i = \lambda_i \mathbf{q}_i` with normalization
    :math:`\mathbf{q}_i^\text{T} \mathbf{q}_i = 1` or
    the generalized eigenvalue problem :math:`\mathbf{A}\mathbf{q}_i = \lambda_i \mathbf{B} \mathbf{q}_i`
    with normalization :math:`\mathbf{q}_i^\text{T} \mathbf{B} \mathbf{q}_i = 1` (:math:`\mathbf{B}` must be positive
    definite). The eigenvectors are returned as a dense matrix ``Q``, where eigenvector ``Q[:, i]`` belongs to
    eigenvalue ``λ[i]``.

    Both real and complex matrices are supported.
    Correct sensitivities for cases with eigenvalue multiplicity are not implemented yet.

    Mode tracking algorithms can be implemented by the user by providing the argument ``sorting_func``, which is a
    function with arguments (``λ``, ``Q``).

    Todo:
        Support for sparse matrix

    Input Signal(s):
      - ``A`` (`dense matrix`): The system matrix of size ``(n, n)``
      - ``B`` (`dense matrix, optional`): Second system matrix (must be positive-definite) of size ``(n, n)``

    Output Signals:
      - ``λ`` (`vector`): Vector with eigenvalues of size ``(n)``
      - ``Q`` (`matrix`): Matrix with eigenvectors ``Q[:, i]`` corresponding to ``λ[i]``, of size ``(n, n)``

    Keyword Args:
        sorting_func: Sorting function to sort the eigenvalues, which must have signature ``func(λ,Q)``
          (default = ``numpy.argsort``)
        hermitian: Flag to omit the automatic detection for Hermitian matrix, saves some work for large matrices
        nmodes: Number of modes to calculate (only for sparse matrices, default = ``0``)
        sigma: Shift value for the eigenvalue problem (only for sparse matrices). Eigenvalues around the shift are
          calculated first (default = ``0.0``)
    """
    def _prepare(self, sorting_func=lambda W, Q: np.argsort(W), hermitian=None, nmodes=None, sigma=None):
        self.sorting_fn = sorting_func
        self.is_hermitian = hermitian
        self.nmodes = nmodes
        self.sigma = sigma
        self.Ainv = None


    def _response(self, A, *args):
        B = args[0] if len(args) > 0 else None
        if self.is_hermitian is None:
            self.is_hermitian = (matrix_is_hermitian(A) and (B is None or matrix_is_hermitian(B)))
        self.is_sparse = sps.issparse(A) and (B is None or sps.issparse(B))
        if self.is_sparse:
            W, Q = self._sparse_eigs(A, B=B)
        else:
            W, Q = spla.eigh(A, b=B) if self.is_hermitian else spla.eig(A, b=B)

        isort = self.sorting_fn(W, Q)
        W = W[isort]
        Q = Q[:, isort]
        for i in range(W.size):
            qi, wi = Q[:, i], W[i]
            qi *= np.sign(np.real(qi[np.argmax(abs(qi)>0)]))
            Bqi = qi if B is None else B@qi
            qi /= np.sqrt(qi@Bqi)
            assert (abs(qi@(qi if B is None else B@qi) - 1.0) < 1e-5)
            assert (np.linalg.norm(A@qi - wi*(qi if B is None else B@qi)) < 1e-5)
        return W, Q

    def _sensitivity(self, dW, dQ):
        A = self.sig_in[0].state
        B = self.sig_in[1].state if len(self.sig_in) > 1 else np.eye(*A.shape)
        dA, dB = None, None
        if not self.is_sparse:
            dA, dB = self._dense_sens(A, B, dW, dQ)
        else:
            if dQ is not None:
                raise NotImplementedError('Sparse eigenvector sensitivities not implemented')
            elif dW is not None:
                dA, dB = self._sparse_eigval_sens(A, B, dW)

        if len(self.sig_in) == 1:
            return dA
        elif len(self.sig_in) == 2:
            return dA, dB


    def _sparse_eigs(self, A, B=None):
        if self.nmodes is None:
            self.nmodes = 6
        if self.sigma is None:
            self.sigma = 0.0

        if self.sigma == 0.0:  # No shift
            mat_shifted = A
        else:
            if B is None:  # If no B is given, use identity to shift
                B = sps.eye(*A.shape)
            mat_shifted = A - self.sigma * B

        # Use shift-and-invert, so make inverse operator
        if self.Ainv is None:
            self.Ainv = auto_determine_solver(mat_shifted, ishermitian=self.is_hermitian)
        self.Ainv.update(mat_shifted)

        AinvOp = spsla.LinearOperator(mat_shifted.shape, matvec=self.Ainv.solve, rmatvec=self.Ainv.adjoint)

        if self.is_hermitian:
            return spsla.eigsh(A, M=B, k=self.nmodes, OPinv=AinvOp, sigma=self.sigma)
        else:
            # TODO
            raise NotImplementedError('Non-Hermitian sparse matrix not supported')

    def _dense_sens(self, A, B, dW, dQ):
        """ Calculates all (eigenvector and eigenvalue) sensitivities for dense matrix """
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

            P = np.block([[(A - wi * B).T, -((B + B.T) / 2) @ qi[..., np.newaxis]], [-B @ qi.T, 0]])
            adj = np.linalg.solve(P, np.block([dqi, dwi]))  # Adjoint solve Lee(1999)
            nu = adj[:-1]
            alpha = adj[-1]
            dAi = np.outer(-nu, qi)
            dA += dAi if np.iscomplexobj(A) else np.real(dAi)
            if dB is not None:
                dBi = np.outer(wi * nu + alpha / 2 * qi, qi)
                dB += dBi if np.iscomplexobj(B) else np.real(dBi)
        return dA, dB

    def _sparse_eigval_sens(self, A, B, dW):
        if dW is None:
            return None, None
        W, Q = [s.state for s in self.sig_out]
        dA, dB = DyadCarrier(), None if B is None else DyadCarrier()
        for i in range(W.size):
            wi, dwi = W[i], dW[i]
            if dwi == 0:
                continue
            qi = Q[:, i]
            qmq = qi@qi if B is None else qi @ (B @ qi)
            dA_u = (dwi/qmq) * qi
            if np.isrealobj(A):
                dA += DyadCarrier([np.real(dA_u), -np.imag(dA_u)], [np.real(qi), np.imag(qi)])
            else:
                dA += DyadCarrier(dA_u, qi)

            if dB is not None:
                dB_u = (wi*dwi/qmq) * qi
                if np.isrealobj(B):
                    dB -= DyadCarrier([np.real(dB_u), -np.imag(dB_u)], [np.real(qi), np.imag(qi)])
                else:
                    dB -= DyadCarrier(dB_u, qi)
        return dA, dB

