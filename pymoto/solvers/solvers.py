import warnings
import numpy as np
from .matrix_checks import matrix_is_hermitian, matrix_is_symmetric, matrix_is_complex


class LinearSolver:
    """ Base class of all linear solvers

    Keyword Args:
        A (matrix): Optionally provide a matrix, which is used in :method:`update` right away.

    Attributes:
        defined (bool): Flag if the solver is able to run, e.g. false if some dependent library is not available
    """

    defined = True
    _err_msg = ""

    def __init__(self, A=None):
        if A is not None:
            self.update(A)

    def update(self, A):
        """ Updates with a new matrix of the same structure

        Args:
            A (matrix): The new matrix of size ``(N, N)``

        Returns:
            self
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def solve(self, rhs, x0=None, trans='N'):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}`

        Args:
            rhs: Right hand side :math:`\mathbf{b}` of shape ``(N)`` or ``(N, K)`` for multiple right-hand-sides
            x0 (optional): Initial guess for the solution
            trans (optional): Option to transpose matrix
                'N':   A   @ x == rhs   (default)   Normal matrix
                'T':   A^T @ x == rhs               Transposed matrix
                'H':   A^H @ x == rhs               Hermitian transposed matrix (conjugate transposed)

        Returns:
            Solution vector :math:`\mathbf{x}` of same shape as :math:`\mathbf{b}`
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    @staticmethod
    def residual(A, x, b, trans='N'):
        r""" Calculates the (relative) residual of the linear system of equations

        The residual is calculated as
        :math:`r = \frac{\left| \mathbf{A} \mathbf{x} - \mathbf{b} \right|}{\left| \mathbf{b} \right|}`

        Args:
            A: The matrix
            x: Solution vector
            b: Right-hand side
            trans (optional): Matrix tranformation (`N` is normal, `T` is transposed, `H` is hermitian transposed)

        Returns:
            Residual value
        """
        if trans == 'N':
            mat = A
        elif trans == 'T':
            mat = A.T
        elif trans == 'H':
            mat = A.conj().T
        else:
            raise TypeError("Only N, T, or H transposition is possible")
        return np.linalg.norm(mat@x - b) / np.linalg.norm(b)


class LDAWrapper(LinearSolver):
    r""" Linear dependency aware solver (LDAS)

    This solver uses previous solutions of the system :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` to reduce computational
    effort. In case the solution :math:`\mathbf{x}` is linearly dependent on the previous solutions, the solution
    will be nearly free of cost.

    Args:
        solver: The internal solver to be used
        tol (optional): Residual tolerance above which the internal solver is used to add a new solution vector.
        A (optional): The matrix :math:`\mathbf{A}`
        symmetric (optional): Flag to indicate a symmetric matrix :math:`A=A^T`
        hermitian (optional): Flag to indicate a Hermitian matrix :math:`A=A^H`

    References:
    Koppen, S., van der Kolk, M., van den Boom, S., & Langelaar, M. (2022).
    Efficient computation of states and sensitivities for compound structural optimisation problems using a Linear Dependency Aware Solver (LDAS).
    Structural and Multidisciplinary Optimization, 65(9), 273.
    DOI: 10.1007/s00158-022-03378-8
    """
    def __init__(self, solver: LinearSolver, tol=1e-7, A=None, symmetric=None, hermitian=None):
        self.solver = solver
        self.tol = tol
        # Storage for solution vectors (solutions of A x = b)
        self.x_stored = []
        self.b_stored = []
        # Storage for adjoint solution vectors (solutions of A^H x = b)
        self.xadj_stored = []
        self.badj_stored = []
        self.A = None
        self._did_solve = False  # For debugging purposes
        self._last_rtol = 0.
        self.hermitian = hermitian
        self.symmetric = symmetric
        self.complex = None
        super().__init__(A)

    def update(self, A):
        """ Clear the internal stored solution vectors and update the internal ``solver`` """
        if self.symmetric is None:
            self.symmetric = matrix_is_symmetric(A)

        if self.hermitian is None:
            if not matrix_is_complex(A):
                self.hermitian = self.symmetric
            self.hermitian = matrix_is_hermitian(A)

        self.A = A
        self.x_stored.clear()
        self.b_stored.clear()
        self.xadj_stored.clear()
        self.badj_stored.clear()
        self.solver.update(A)

    def _do_solve_1rhs(self, A, rhs, x_data, b_data, solve_fn, x0=None):
        dtype = np.result_type(A, rhs)
        rhs_loc = np.zeros_like(rhs, dtype=dtype)
        rhs_loc[:] = rhs
        sol = np.zeros_like(rhs_loc, dtype=dtype)

        # Check linear dependencies in the rhs using modified Gram-Schmidt
        for (x, b) in zip(x_data, b_data):
            alpha = rhs_loc @ b.conj() / (b.conj() @ b)
            rem_rhs = alpha*b
            add_sol = alpha*x

            if (np.iscomplexobj(add_sol) or np.iscomplexobj(rem_rhs)) and not (np.iscomplexobj(sol) or np.iscomplexobj(rhs_loc)):
                small_imag_rhs = np.linalg.norm(np.imag(rem_rhs)) < 1e-10*np.linalg.norm(np.real(rem_rhs))
                small_imag_sol = np.linalg.norm(np.imag(add_sol)) < 1e-10*np.linalg.norm(np.real(add_sol))
                if small_imag_rhs and small_imag_sol:
                    rem_rhs = np.real(rem_rhs)
                    add_sol = np.real(add_sol)
                else:
                    # Complex vector cannot be added to real solution
                    continue
            rhs_loc -= rem_rhs
            sol += add_sol
            if x0 is not None:
                # Remove solution from x0 vector
                beta = x0 @ x.conj() / (x.conj() @ x)
                x0 -= beta * x

        # Check tolerance
        self._last_rtol = 1.0 if len(x_data) == 0 else self.residual(A, sol, rhs)

        if self._last_rtol > self.tol:
            # Calculate a new solution
            xnew = solve_fn(rhs_loc, x0)
            x_data.append(xnew)
            bnew = np.zeros_like(rhs)
            bnew[:] = A@xnew
            b_data.append(bnew)
            sol += xnew
            self._did_solve = True
        else:
            self._did_solve = False

        return sol

    def _solve_1x(self, b, storage='N', x0=None):
        if storage == 'N':  # Use the normal storage
            return self._do_solve_1rhs(self.A, b, self.x_stored, self.b_stored,
                                       lambda rhs, x0: self.solver.solve(rhs, trans='N', x0=x0), x0=x0)
        elif storage == 'H':  # Use adjoint storage
            return self._do_solve_1rhs(self.A.conj().T, b, self.xadj_stored, self.badj_stored,
                                       lambda rhs, x0: self.solver.solve(rhs, trans='H', x0=x0), x0=x0)

    def solve(self, rhs, x0=None, trans='N'):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by performing a modified
        Gram-Schmidt over the previously calculated solutions :math:`\mathbf{U}` and corresponding right-hand-sides
        :math:`\mathbf{F}`. This is used to construct an approximate solution
        :math:`\tilde{\mathbf{x}} = \sum_k \alpha_k \mathbf{u}_k` in the subspace of :math:`\mathbf{U}`.
        If the residual of :math:`\mathbf{A} \tilde{\mathbf{x}} = \mathbf{b}` is above the tolerance, a new solution
        :math:`\mathbf{u}_{k+1}` will be added to the database such that
        :math:`\mathbf{x} = \tilde{\mathbf{x}}+\mathbf{u}_{k+1}` is the solution to the system
        :math:`\mathbf{A} \mathbf{x} = \mathbf{b}`.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.
        """
        ''' Required    Symm. matrix    Herm. matrix    Any matrix (uses adjoint storage)
                        A^T = A         A^H = A
            A x = b
            A^T x = b   A x = b         A x^* = b^*     A^H x^* = b^*
            A^H x = b   A x^* = b^*     A x = b         A^H x = b
            
            For symmetric or Hermitian matrices, only the normal storage is required. For any other matrix, the `T` and 
            `H` mode will require the adjoint storage space.
        '''
        if trans not in ['N', 'T', 'H']:
            raise TypeError("Only N, T, or H transposition is possible")

        # Use adjoint storage?
        adjoint_mode = trans != 'N' and not (self.symmetric or self.hermitian)
        # Use conjugation?
        conj_mode = self.symmetric and trans == 'H' or not self.symmetric and trans == 'T'

        storage = 'H' if adjoint_mode else 'N'
        rhs = rhs.conj() if conj_mode else rhs

        if rhs.ndim == 1:
            ret = self._solve_1x(rhs, storage=storage, x0=x0)
        else:  # Multiple rhs
            if x0 is not None and x0.shape != rhs.shape:
                warnings.warn(f'Shape of x0 {x0.shape}, but expected {rhs.shape}. Not using initial guess.')
                x0 = None
            sol = []
            for i in range(rhs.shape[-1]):
                sol.append(self._solve_1x(rhs[..., i], storage=storage, x0=x0[..., i] if x0 is not None else None))
            ret = np.stack(sol, axis=-1)
        return ret.conj() if conj_mode else ret
