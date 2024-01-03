""" Specialized linear algebra modules """
import warnings
from inspect import currentframe, getframeinfo

import numpy as np
import scipy.linalg as spla  # Dense matrix solvers
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from pymoto import Signal, Module, DyadCarrier, LDAWrapper
from pymoto import SolverDenseLU, SolverDenseLDL, SolverDenseCholesky, SolverDiagonal, SolverDenseQR
from pymoto import SolverSparseLU, SolverSparseCholeskyCVXOPT, SolverSparsePardiso, SolverSparseCholeskyScikit
from pymoto import matrix_is_symmetric, matrix_is_hermitian, matrix_is_diagonal


class StaticCondensation(Module):
    r"""Static condensation of a linear system of equations

    The partitioned system of equations

    :math:`\begin{bmatrix} \mathbf{A}_\text{mm} & \mathbf{A}_\text{ms} \\ \mathbf{A}_\text{sm} & \mathbf{A}_\text{ss}
    \end{bmatrix}
    \begin{bmatrix} \mathbf{x}_\text{m} \\ \mathbf{x}_\text{s} \end{bmatrix} =
    \begin{bmatrix} \mathbf{b}_\text{m} \\ \mathbf{b}_\text{s} \end{bmatrix}
    ,`
    with subscripts ``(m)`` and ``(s)`` referring to the main and secondary dofs, respectively.

    The system is solved in two steps:

    :math:`\begin{aligned}
    \mathbf{A}_\text{ss} \mathbf{x}_\text{sm} &= \mathbf{A}_\text{sm} \\
    \tilde{\mathbf{A}} &= \mathbf{A}_\text{mm} - \mathbf{A}_\text{ms} \mathbf{x}_\text{sm}.
    \end{aligned}`

    Assumptions:
        (i) It is assumed the prescribed DOFs (all dof - main dof - free dof) are prescribed to zero.
        (ii) It is assumed the applied load on the free DOFs is zero; there is no reduced load.

    Implemented by @artofscience (s.koppen@tudelft.nl).

    References:

    Koppen, S., Langelaar, M., & van Keulen, F. (2022).
    Efficient multi-partition topology optimization.
    Computer Methods in Applied Mechanics and Engineering, 393, 114829.
    DOI: https://doi.org/10.1016/j.cma.2022.114829

    Input Signals:
      - ``A`` (`dense or sparse matrix`): The system matrix :math:`\mathbf{A}` of size ``(n, n)``

    Output Signal:
      - ``Ared`` (`dense or sparse matrix`): The reduced system matrix :math:`\tilde{\mathbf{A}}` of size ``(m, m)``

    Args:
        free: The indices corresponding to the free degrees of freedom
        main: The indices corresponding to the main degrees of freedom
        **kwargs: See `pymoto.LinSolve`, as they are directly passed into the `LinSolve` module
    """

    def _prepare(self, main, free, **kwargs):
        self.module_LinSolve = LinSolve([self.sig_in[0], Signal()], **kwargs)
        self.module_LinSolve.use_lda_solver = False
        self.m = main
        self.f = free

    def _response(self, A):
        self.n = np.shape(A)[0]
        self.module_LinSolve.sig_in[0].state = A[self.f, ...][..., self.f]
        self.module_LinSolve.sig_in[1].state = A[self.f, ...][..., self.m].todense()
        self.module_LinSolve.response()
        self.X = self.module_LinSolve.sig_out[0].state
        return A[self.m, ...][..., self.m] - A[self.m, ...][..., self.f] @ self.X

    def _sensitivity(self, dfdB):
        C = np.zeros((self.n, len(self.m)), dtype=float)
        C[self.m, ...] = np.eye(len(self.m))
        C[self.f, ...] = -self.X
        return C @ dfdB @ C.T if isinstance(dfdB, DyadCarrier) else DyadCarrier(list(C.T), list(np.asarray(dfdB @ C.T)))


class SystemOfEquations(Module):
    r""" Solve a partitioned linear system of equations

    The partitioned system of equations

    :math:`\begin{bmatrix} \mathbf{A}_\text{ff} & \mathbf{A}_\text{fp} \\ \mathbf{A}_\text{pf} & \mathbf{A}_\text{pp}
    \end{bmatrix}
    \begin{bmatrix} \mathbf{x}_\text{f} \\ \mathbf{x}_\text{p} \end{bmatrix} =
    \begin{bmatrix} \mathbf{b}_\text{f} \\ \mathbf{b}_\text{p} \end{bmatrix}
    ,`

    which is solved in two steps. First solving for the free unknowns (e.g. displacements or temperatures)
    :math:`\mathbf{x}_f` and then calculating the rhs for the prescribed unknowns (e.g. reaction forces or heat flux):

    :math:`\begin{aligned}
    \mathbf{A}_\text{ff} \mathbf{x}_\text{f} &= \mathbf{b}_\text{f} - \mathbf{A}_\text{fp} \mathbf{x}_\text{p} \\
    \mathbf{b}_\text{p} &= \mathbf{A}_\text{pf}\mathbf{x}_\text{f} + \mathbf{A}_\text{pp} \mathbf{x}_\text{p}.
    \end{aligned}`

    References:
        Koppen, S., Langelaar, M., & van Keulen, F. (2022).
        Efficient multi-partition topology optimization.
        Computer Methods in Applied Mechanics and Engineering, 393, 114829.
        DOI: https://doi.org/10.1016/j.cma.2022.114829

    Input Signals:
      - ``A`` (`dense or sparse matrix`): The system matrix :math:`\mathbf{A}` of size ``(n, n)``
      - ``b_f`` (`vector`): applied load vector of size ``(f)`` or block-vector of size ``(f, Nrhs)``
      - ``x_p`` (`vector`): prescribed state vector of size ``(p)`` or block-vector of size ``(p, Nrhs)``

    Output Signal:
      - ``x`` (`vector`): state vector of size ``(n)`` or block-vector of size ``(n, Nrhs)``
      - ``b`` (`vector`): load vector of size ``(n)`` or block-vector of size ``(n, Nrhs)``

    Keyword Args:
        free: The indices corresponding to the free degrees of freedom at which :math:`f_\text{f}` is given
        prescribed: The indices corresponding to the prescibed degrees of freedom at which :math:`x_\text{p}` is given
        **kwargs: See `pymoto.LinSolve`, as they are directly passed into the `LinSolve` module
    """

    def _prepare(self, free=None, prescribed=None, **kwargs):
        self.module_LinSolve = LinSolve([self.sig_in[0], Signal()], **kwargs)
        self.f = free
        self.p = prescribed
        assert self.p is not None or self.f is not None, "Either prescribed or free indices must be provided"

    def _response(self, A, bf, xp):
        assert bf.shape[0] + xp.shape[0] == A.shape[0], "Dimensions of applied force and displacement must match matrix"
        assert bf.ndim == xp.ndim, "Number of loadcases for applied force and displacement must match"
        self.n = np.shape(A)[0]
        self.dim = xp.ndim

        if self.f is None:
            all_dofs = np.arange(self.n)
            self.f = np.setdiff1d(all_dofs, self.p)
        if self.p is None:
            all_dofs = np.arange(self.n)
            self.p = np.setdiff1d(all_dofs, self.f)
        assert self.f.size + self.p.size == self.n, "Size of free and prescribed indices must match the matrix size"

        # create empty output
        self.x = np.zeros((self.n, *bf.shape[1:]), dtype=complex) if np.iscomplexobj(A) else np.zeros(
            (self.n, *bf.shape[1:]), dtype=float)
        self.x[self.p, ...] = xp

        b = np.zeros_like(self.x)
        b[self.f, ...] = bf

        # partitioning
        Aff = A[self.f, :][:, self.f]
        self.Afp = A[self.f, :][:, self.p]
        self.App = A[self.p, :][:, self.p]

        # solve
        self.module_LinSolve.sig_in[0].state = Aff
        self.module_LinSolve.sig_in[1].state = bf - self.Afp * xp
        self.module_LinSolve.response()
        xf = self.module_LinSolve.sig_out[0].state

        # set output
        self.x[self.f, ...] = xf
        b[self.p, ...] = self.Afp.T * xf + self.App * xp

        return self.x, b

    def _sensitivity(self, dgdx, dgdb):
        adjoint_load = np.zeros_like(self.x[self.f, ...])

        if dgdx is not None:
            adjoint_load += dgdx[self.f, ...]
        if dgdb is not None:
            adjoint_load += self.Afp * dgdb[self.p, ...]

        lam = np.zeros_like(self.x)
        lamf = -1.0 * self.module_LinSolve.solver.adjoint(adjoint_load)
        lam[self.f, ...] = lamf

        if dgdb is not None:
            lam[self.p, ...] = dgdb[self.p, ...]

        # sensitivities to system matrix
        if self.x.ndim > 1:
            dgdA = DyadCarrier(list(lam.T), list(self.x.T))
        else:
            dgdA = DyadCarrier(lam, self.x)

        # sensitivities to applied load and prescribed state
        dgdbf = np.zeros_like(adjoint_load)
        dgdup = np.zeros_like(self.x[self.p, ...])
        dgdbf -= lam[self.f, ...]
        dgdup += self.Afp.T * lam[self.f, ...]

        if dgdx is not None:
            dgdup += dgdx[self.p, ...]

        if dgdb is not None:
            dgdbf += dgdb[self.f, ...]
            dgdup += self.App * dgdb[self.p, ...]

        return dgdA, dgdbf, dgdup


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
            # TODO check for positive definiteness?  np.all(A.diagonal() > 0) or np.all(A.diagonal() < 0)
            return SolverSparsePardiso(symmetric=issymmetric, hermitian=ishermitian, positive_definite=ispositivedefinite)

        if ishermitian:
            # Check if diagonal is all positive or all negative -> Cholesky
            if np.all(A.diagonal() > 0) or np.all(A.diagonal() < 0):  # TODO what about the complex case?
                if SolverSparseCholeskyScikit.defined:
                    return SolverSparseCholeskyScikit()
                if SolverSparseCholeskyCVXOPT.defined:
                    return SolverSparseCholeskyCVXOPT()

        return SolverSparseLU()  # Default to LU, which should be possible for any non-singular square matrix

    else:  # Dense branch
        if ishermitian:
            # Check if diagonal is all positive or all negative
            if np.all(A.diagonal() > 0) or np.all(A.diagonal() < 0):
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
      - ``b`` (`vector`): Right-hand-side vector of size ``(n)`` or block-vector of size ``(n, Nrhs)``

    Output Signal:
      - ``x`` (`vector`): Solution vector of size ``(n)`` or block-vector of size ``(n, Nrhs)``

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
            raise TypeError("Complex right-hand-side for a real-valued sparse matrix is not supported."
                            "This case can simply be solved by running two rhs (one for the real part and "
                            "one for the imaginary.")

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
            if self.u.ndim > 1:
                dmat = DyadCarrier(list(-lam.T), list(self.u.T))
            else:
                dmat = DyadCarrier(-lam, self.u)
        else:
            if self.u.ndim > 1:
                dmat = np.einsum("iB,jB->ij", -lam, self.u, optimize=True)
            else:
                dmat = np.outer(-lam, self.u)
        if not self.iscomplex:
            dmat = dmat.real

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
        mode: Mode of the eigensolver (see documentation of `scipy.sparse.linalg.eigsh` for more info)
    """
    def _prepare(self, sorting_func=lambda W, Q: np.argsort(W), hermitian=None, nmodes=None, sigma=None, mode='normal'):
        self.sorting_fn = sorting_func
        self.is_hermitian = hermitian
        self.nmodes = nmodes
        self.sigma = sigma
        self.mode = mode
        self.Ainv = None

    def _response(self, A, *args):
        B = args[0] if len(args) > 0 else None
        if self.is_hermitian is None:
            self.is_hermitian = (matrix_is_hermitian(A) and (B is None or matrix_is_hermitian(B)))
        self.is_sparse = sps.issparse(A) and (B is None or sps.issparse(B))

        # Solve the eigenvalue problem
        if self.is_sparse:
            W, Q = self._sparse_eigs(A, B=B)
        else:
            W, Q = spla.eigh(A, b=B) if self.is_hermitian else spla.eig(A, b=B)

        # Sort the eigenvalues
        isort = self.sorting_fn(W, Q)
        W = W[isort]
        Q = Q[:, isort]

        # Normalize the eigenvectors
        for i in range(W.size):
            qi, wi = Q[:, i], W[i]
            qi *= np.sign(np.real(qi[np.argmax(abs(qi) > 0)]))  # Set first value positive for orientation
            Bqi = qi if B is None else B@qi
            qi /= np.sqrt(qi@Bqi)  # Normalize
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
            return spsla.eigsh(A, M=B, k=self.nmodes, OPinv=AinvOp, sigma=self.sigma, mode=self.mode)
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

