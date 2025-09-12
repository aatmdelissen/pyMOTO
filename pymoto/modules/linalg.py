"""Specialized linear algebra modules"""

import warnings
from typing import Callable

import numpy as np
import scipy.linalg as spla  # Dense matrix solvers
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

from pymoto import Signal, Module, DyadCarrier
from pymoto.solvers import auto_determine_solver, LinearSolver
from pymoto.solvers import matrix_is_sparse, matrix_is_complex, matrix_is_hermitian, LDAWrapper


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

    def __init__(self, main, free, **kwargs):
        """Initialize the static condensation module

        Args:
            main: The indices corresponding to the free degrees of freedom
            free: The indices corresponding to the main degrees of freedom
            **kwargs: Are passed to :py:class:`pymoto.LinSolve` module
        """
        self.m_linsolve = LinSolve(**kwargs)
        self.m_linsolve.use_lda_solver = False
        self.m = main
        self.f = free  # TODO free dofs can be deduced from main? Although a third set of fixed dofs may be required.

    def __call__(self, A):
        self.n = np.shape(A)[0]

        Aff = A[self.f, ...][..., self.f]
        Afm = A[self.f, ...][..., self.m].todense()
        Amm = A[self.m, ...][..., self.m]
        Amf = A[self.m, ...][..., self.f]

        if self.m_linsolve.sig_in is None:
            sAff = Signal("Aff", state=Aff)
            sAfm = Signal("rhs", state=Afm)
            self.m_linsolve.connect([sAff, sAfm])
        else:
            self.m_linsolve.sig_in[0].state = Aff
            self.m_linsolve.sig_in[1].state = Afm
            self.m_linsolve.response()
        self.X = self.m_linsolve.sig_out[0].state
        return Amm - Amf @ self.X

    def _sensitivity(self, dfdB):
        C = np.zeros((self.n, len(self.m)), dtype=float)
        C[self.m, ...] = np.eye(len(self.m))
        C[self.f, ...] = -self.X
        return (
            C @ dfdB @ C.T if isinstance(dfdB, DyadCarrier) else DyadCarrier(list(C.T), list(np.asarray(dfdB @ C.T)))
        )  # FIXME probably can do without the check


class Inverse(Module):
    r"""Calculate the inverse of a matrix :math:`\mathbf{B} = \mathbf{A}^{-1}`

    Input Signal:
      - ``A`` (`np.ndarray`): Dense matrix:math:`\mathbf{A}`

    Output Signal:
      - `B` (`np.ndarray`): The inverse of :math:`\mathbf{A}` as dense matrix
    """

    def __call__(self, A):
        return np.linalg.inv(A)

    def _sensitivity(self, dB):
        A = self.sig_in[0].state
        B = self.sig_out[0].state
        dA = -B.T @ dB @ B.T
        return dA if np.iscomplexobj(A) else np.real(dA)


class LinSolve(Module):
    r"""Solves linear system of equations :math:`\mathbf{A}\mathbf{x}=\mathbf{b}`

    Self-adjointness is automatically detected using :class:`LDAWrapper`.

    Input Signals:
      - ``A`` (`dense or sparse matrix`): The system matrix :math:`\mathbf{A}` of size ``(n, n)``
      - ``b`` (`vector`): Right-hand-side vector of size ``(n)`` or block-vector of size ``(n, Nrhs)``

    Output Signal:
      - ``x`` (`vector`): Solution vector of size ``(n)`` or block-vector of size ``(n, Nrhs)``

    Attributes:
        use_lda_solver: Use the linear-dependency-aware solver :class:`LDAWrapper` to prevent redundant computations
    """

    use_lda_solver = True

    def __init__(self, dep_tol: float = 1e-5, 
                 hermitian: bool = None, 
                 symmetric: bool = None, 
                 positive_definite: bool = None,
                 solver: LinearSolver = None
                 ):
        """Initialize the linear solver module

        Args:
            dep_tol (float, optional): Tolerance for detecting linear dependence of solution vectors. Defaults to 1e-5.
            hermitian (bool, optional): Flag to omit the automatic detection for Hermitian matrix, saves some work for 
              large matrices
            symmetric (bool, optional): Flag to omit the automatic detection for symmetric matrix, saves some work for 
              large matrices
            positive_definite (bool, optional): Flag to specify if the matrix is positive definite
            solver (:py:class:`pymoto.solvers.LinearSolver`, optional): Manually override the linear solver used, 
              instead of the the solver from :func:`pymoto.solvers.auto_determine_solver`
        """
        self.dep_tol = dep_tol
        self.ishermitian = hermitian
        self.issymmetric = symmetric
        self.ispositivedefinite = positive_definite
        self.solver = solver
        self.u = None  # Solution storage

    def __call__(self, mat, rhs):
        # Do some detections on the matrix type
        self.issparse = matrix_is_sparse(mat)  # Check if it is a sparse matrix
        self.iscomplex = matrix_is_complex(mat)  # Check if it is a complex-valued matrix
        if not self.iscomplex and self.issymmetric is not None:
            self.ishermitian = self.issymmetric
        if self.ishermitian is None:
            self.ishermitian = matrix_is_hermitian(mat)
        if self.issparse and not self.iscomplex and np.iscomplexobj(rhs):
            raise TypeError(
                "Complex right-hand-side for a real-valued sparse matrix is not supported."
                "This case can simply be solved by running two rhs (one for the real part and "
                "one for the imaginary."
            )  # FIXME

        # Determine the solver we want to use
        if self.solver is None:
            self.solver = auto_determine_solver(mat, 
                                                ishermitian=self.ishermitian, 
                                                issymmetric=self.issymmetric, 
                                                ispositivedefinite=self.ispositivedefinite)
        if not isinstance(self.solver, LDAWrapper) and self.use_lda_solver:
            lda_kwargs = dict(hermitian=self.ishermitian, symmetric=self.issymmetric)
            if hasattr(self.solver, "tol"):
                lda_kwargs["tol"] = self.solver.tol * 5
            self.solver = LDAWrapper(self.solver, **lda_kwargs)

        # Update solver with new matrix
        self.solver.update(mat)

        # Solution
        self.u = self.solver.solve(rhs, x0=self.u)

        return self.u

    def _sensitivity(self, dfdv):
        mat, rhs = self.get_input_states()
        # lam = self.solver.solve(dfdv.conj(), trans='H').conj()
        lam = self.solver.solve(dfdv, trans="T")

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
    """

    def __init__(self, free=None, prescribed=None, **kwargs):
        """Initialize the system of equations module

        The free indices, prescribed indices, or both  must be provided.

        Args:
            free (optional): The indices corresponding to the free degrees of freedom at which :math:`f_\text{f}` is 
              given
            prescribed (optional): The indices corresponding to the prescibed degrees of freedom at which 
              :math:`x_\text{p}` is given
            **kwargs: Arguments passed to initialization of :py:class:`pymoto.LinSolve`
        """
        self.linsolve = LinSolve(**kwargs)
        self.linsolve.sig_in = [Signal("Aff"), Signal("bf - Afp.xp")]
        self.linsolve.sig_out = [Signal("xf")]
        self.f = free
        self.p = prescribed
        if self.p is None and self.f is None:
            raise ValueError("Either prescribed or free indices must be provided")

    def __call__(self, A, bf, xp):
        n = A.shape[0]

        assert bf.shape[0] + xp.shape[0] == n, "Dimensions of applied force and displacement must match matrix"
        assert bf.ndim == xp.ndim, "Number of loadcases for applied force and displacement must match"

        if self.f is None:
            all_dofs = np.arange(n)
            self.f = np.setdiff1d(all_dofs, self.p)
        if self.p is None:
            all_dofs = np.arange(n)
            self.p = np.setdiff1d(all_dofs, self.f)
        assert self.f.size + self.p.size == n, "Size of free and prescribed indices must match the matrix size"

        # create empty output
        self.x = np.zeros((n, *bf.shape[1:]), dtype=A.dtype)
        self.x[self.p, ...] = xp

        b = np.zeros_like(self.x)
        b[self.f, ...] = bf

        # partitioning
        Aff = A[self.f, :][:, self.f]
        self.Afp = A[self.f, :][:, self.p]
        self.App = A[self.p, :][:, self.p]

        # solve
        self.linsolve.sig_in[0].state = Aff
        self.linsolve.sig_in[1].state = bf - self.Afp * xp
        self.linsolve.response()
        xf = self.linsolve.sig_out[0].state

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
        lamf = -1.0 * self.linsolve.solver.solve(adjoint_load, trans="T")
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


class EigenSolve(Module):
    r"""Solves the (generalized) eigenvalue problem :math:`\mathbf{A}\mathbf{q}_i = \lambda_i \mathbf{B} \mathbf{q}_i`

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

    Input Signal(s):
      - ``A`` (`dense matrix`): The system matrix of size ``(n, n)``
      - ``B`` (`dense matrix, optional`): Second system matrix (must be positive-definite) of size ``(n, n)``

    Output Signals:
      - ``λ`` (`vector`): Vector with eigenvalues of size ``(n)``
      - ``Q`` (`matrix`): Matrix with eigenvectors ``Q[:, i]`` corresponding to ``λ[i]``, of size ``(n, n)``
    """

    def __init__(self, 
                 sorting_func: Callable = (lambda W, Q: np.argsort(W)), 
                 hermitian: bool = None, 
                 nmodes: int = None, 
                 sigma: complex = None, 
                 mode: str = "normal"):
        """Initialize the eigenvalue solver module

        Args:
            sorting_func (Callable, optional): Sorting function to sort the eigenvalues, which must have signature 
              ``func(λ,Q)``. Defaults to (lambda W, Q: np.argsort(W)).
            hermitian (bool, optional): Flag to omit the automatic detection for Hermitian matrix, saves some work for 
              large matrices
            nmodes (int, optional): Number of modes to calculate (only for sparse matrices)
            sigma (complex, optional): Shift value for the shift-and-invert eigenvalue problem (only for sparse 
              matrices). Eigenvalues around the shift are calculated first. Defaults to None.
            mode (str, optional): Mode of the eigensolver (see documentation of `scipy.sparse.linalg.eigsh` for more 
              info). Defaults to "normal".
        """
        self.sorting_fn = sorting_func
        self.is_hermitian = hermitian
        self.nmodes = nmodes
        self.sigma = sigma
        self.mode = mode
        self.Ainv = None
        self.do_solve = False
        self.adjoint_solvers_need_update = True
        self.dense_solver = None

    def __call__(self, A, *args):
        B = args[0] if len(args) > 0 else None
        if self.is_hermitian is None:
            self.is_hermitian = matrix_is_hermitian(A) and (B is None or matrix_is_hermitian(B))
        self.is_sparse = matrix_is_sparse(A) and (B is None or matrix_is_sparse(B))
        self.adjoint_solvers_need_update = True

        # Solve the eigenvalue problem
        if self.is_sparse:
            W, Q = self._sparse_eigs(A, B=B)
        else:
            if self.dense_solver is None:
                self.dense_solver = spla.eigh if self.is_hermitian else spla.eig
            try:
                W, Q = self.dense_solver(A, b=B)
            except np.linalg.LinAlgError:  # eigh fails for non-positive definite B
                if not self.dense_solver == spla.eigh:
                    raise
                W, Q = spla.eig(A, b=B)
                self.dense_solver = spla.eig
                print("Failed before and switched to eig -- succesfully")

        # Sort the eigenvalues
        isort = self.sorting_fn(W, Q)
        W = W[isort]
        Q = Q[:, isort]

        # Normalize the eigenvectors with the following conditions:
        # 1) Flip sign such that the average (real) value is positive
        # 2) Normalize the eigenvector v⋅v or v⋅Bv to unity
        for i in range(W.size):
            # wi = W[i]
            qi = Q[:, i]
            Bqi = qi if B is None else B @ qi

            normval = np.sqrt(qi @ Bqi)
            sgn = 1 if np.real(np.average(qi)) >= 0 else -1
            sf = sgn / normval
            assert np.isfinite(sf)
            if sf == 0.0:
                warnings.warn(f"Scaling factor of mode {i} is zero!")
            qi *= sf
        return W, Q

    def _sensitivity(self, dW, dQ):
        A = self.sig_in[0].state
        B = self.sig_in[1].state if len(self.sig_in) > 1 else np.eye(*A.shape)
        dA, dB = None, None
        if not self.is_sparse:
            dA, dB = self._dense_sens(A, B, dW, dQ)
        else:
            if dQ is not None:
                # raise NotImplementedError('Sparse eigenvector sensitivities not implemented')
                dA, dB = self._sparse_eigvec_sens(A, B, dW, dQ)
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
            self.do_solve = True
        if self.sigma != 0:
            self.do_solve = True
        if self.do_solve:
            self.Ainv.update(mat_shifted)

        AinvOp = spsla.LinearOperator(
            mat_shifted.shape, matvec=self.Ainv.solve, rmatvec=lambda b: self.Ainv.solve(b, trans="H")
        )

        if self.is_hermitian:
            return spsla.eigsh(A, M=B, k=self.nmodes, OPinv=AinvOp, sigma=self.sigma, mode=self.mode)
        else:
            if self.mode.lower() not in ["normal"]:
                raise NotImplementedError("Only `normal` mode can be selected for non-hermitian matrix")
            return spsla.eigs(A, M=B, k=self.nmodes, OPinv=AinvOp, sigma=self.sigma)

    def _dense_sens(self, A, B, dW, dQ):
        """Calculates all (eigenvector and eigenvalue) sensitivities for dense matrix"""
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
            qmq = qi @ qi if B is None else qi @ (B @ qi)
            dA_u = (dwi / qmq) * qi
            dAi = DyadCarrier(dA_u, qi)
            dA += np.real(dAi) if np.isrealobj(A) else dAi

            if dB is not None:
                dB_u = (wi * dwi / qmq) * qi
                dBi = DyadCarrier(dB_u, qi)
                dB -= np.real(dBi) if np.isrealobj(B) else dBi

        return dA, dB

    def _sparse_eigvec_sens(self, A, B, dW, dQ):
        """Calculate eigenvector sensitivities for a sparse eigenvalue problem
        References:
             Delissen (2022), Topology optimization for dynamic and controlled systems,
               doi: https://doi.org/10.4233/uuid:c9ed8f61-efe1-4dc8-bb56-e353546cf247

        Args:
            A: System matrix
            B: Mass matrix
            dW: Adjoint eigenvalue sensitivities
            dQ: Adjoint eigenvector sensitivities

        Returns:
            dA: Adjoint system matrix sensitivities
            dB: Adjoint mass matrix sensitivities
        """
        if dQ is None:
            return self._sparse_eigval_sens(A, B, dW)
        W, Q = [s.state for s in self.sig_out]
        if dW is not None:
            dA, dB = self._sparse_eigval_sens(A, B, dW)
        else:
            dA, dB = DyadCarrier(), None if B is None else DyadCarrier()
        for i in range(W.size):
            phi = Q[:, i]
            dphi = dQ[:, i]
            if dphi.min() == dphi.max() == 0.0:
                continue
            lam = W[i]

            alpha = -phi @ dphi
            r = dphi + alpha * B.T @ phi

            # Solve particular solution
            if self.adjoint_solvers_need_update or self.solvers[i] is None:
                Z = A - lam * B
                if not hasattr(self, "solvers"):
                    self.solvers = [None for _ in range(W.size)]
                if self.solvers[i] is None:  # Solver must be able to solve indefinite system
                    self.solvers[i] = auto_determine_solver(Z, ispositivedefinite=False)
                if self.adjoint_solvers_need_update:
                    self.solvers[i].update(Z)

            vp = self.solvers[i].solve(r, trans="T")

            # Calculate total ajoint by adding homogeneous solution
            c = -vp @ B @ phi
            v = vp + c * phi

            # Add to mass and stiffness matrix
            dAi = -DyadCarrier(v, phi)
            dA += np.real(dAi) if np.isrealobj(A) else dAi
            if B is not None:
                dBi = DyadCarrier(alpha / 2 * phi + lam * v, phi)
                dB += np.real(dBi) if np.isrealobj(B) else dBi
        return dA, dB
