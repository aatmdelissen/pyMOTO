import warnings
import numpy as np
import scipy.sparse as sps
from scipy.sparse import SparseEfficiencyWarning
from .solvers import matrix_is_hermitian, matrix_is_complex, matrix_is_symmetric, LinearSolver

# ------------------------------------ Pardiso Solver -----------------------------------
try:
    from pypardiso import PyPardisoSolver
    _has_pardiso = True
except ImportError:
    _has_pardiso = False


class SolverSparsePardiso(LinearSolver):
    """ Solver wrapper Intel MKL Pardiso solver, which is a very fast and flexible multi-threaded solver

    Complex-valued matrices are currently not supported

    Uses the Python interface ``pypardiso`` to the Intel MKL PARDISO library for solving large sparse linear systems of
    equations Ax=b.

    Args:
        A (optional): The matrix
        symmetric (optional): If it is already known if the matrix is symmetric, you can provide it here
        hermitian (optional): If it is already known if the matrix is Hermitian, you can provide it here
        positive_definite (optional): If positive-definiteness is known, provide it here

    References:
        `PyPardiso <https://github.com/haasad/PyPardisoProject>`_
        `Intel MKL Pardiso <https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/
          top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface.html>`_
    """

    defined = _has_pardiso

    def __init__(self, A=None, symmetric=None, hermitian=None, positive_definite=None):
        super().__init__(A)
        if not self.defined:
            raise ImportError("Intel MKL Pardiso solver (pypardiso) cannot be found. ")

        self.mtype = None
        self.hermitian = hermitian
        self.symmetric = symmetric
        self.positive_definite = positive_definite
        self._mtype = None
        self._pardiso_solver = None

    def _determine_mtype(self, A):
        """ Determines the matrix-type according to Intel

        ``mtype``: Matrix type
          - ``1`` Real and structurally symmetric
          - ``2`` Real and symmetric positive definite
          - ``-2`` Real and symmetric indefinite
          - ``3`` Complex and structurally symmetric
          - ``4`` Complex and Hermitian positive definite
          - ``-4`` Complex and Hermitian indefinite
          - ``6`` Complex and symmetric
          - ``11`` Real and non-symmetric
          - ``13`` Complex and non-symmetric

        References:
            `Intel MKL Pardiso arguments <https://www.intel.com/content/www/us/en/develop/documentation/
              onemkl-developer-reference-c/top/sparse-solver-routines/
              onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso.html#pardiso>`_

        Args:
            A: The matrix

        Return:
            Intel MKL mtype
        """
        # TODO structural symmetry is not checked, may be added at some point
        self.complex = matrix_is_complex(A)
        self.dtype = np.complex128 if self.complex else np.float64
        if self.complex:
            # Check if is Hermitian?
            if self.hermitian is None:
                self.hermitian = matrix_is_hermitian(A)
            if self.hermitian:
                return 4 if self.positive_definite else -4  # TODO Works if 13 is used

            # Is the matrix symmetric?
            if self.symmetric is None:
                self.symmetric = matrix_is_symmetric(A)
            return 6 if self.symmetric else 13
        else:
            # Determine symmetry
            if self.symmetric is None:
                self.symmetric = self.hermitian if self.hermitian is not None else matrix_is_symmetric(A)
            if self.symmetric:
                return 2 if self.positive_definite else -2
            return 11

    def update(self, A):
        """ Factorize the matrix A, the factorization will automatically be used if the same matrix A is passed to the
        solve method. This will drastically increase the speed of solve, if solve is called more than once for the
        same matrix A

        Args:
            A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix)
        """
        # Update matrix type
        if self._mtype is None:
            self._mtype = self._determine_mtype(A)

        if self._mtype in {-2, 2, 6}:
            A = sps.triu(A, format='coo')  # Only use upper part
            # Explicitly set zero diagonal entries, as this is better for Intel Pardiso
            zero_diag_entries, = np.where(A.diagonal() == 0)
            if len(zero_diag_entries) > 0:
                A.row = np.append(A.row, zero_diag_entries)
                A.col = np.append(A.col, zero_diag_entries)
                A.data = np.append(A.data, np.zeros_like(zero_diag_entries))
            A = A.tocsr()
            # assert(np.diff(A.indptr).all())  # Every row must be non-empty
            # assert(max(np.diff(A.indices[A.indptr[:-1]])))  # All diagonals must be non-empty
        if not sps.isspmatrix_csr(A):
            warnings.warn(f'PyPardiso requires CSR matrix format, not {type(A).__name__}', SparseEfficiencyWarning)
            A = A.tocsr()
        self.A = A

        if self._pardiso_solver is None:
            self._pardiso_solver = PyPardisoSolver(mtype=self._mtype)

        self._pardiso_solver.factorize(A)

    def solve(self, b):
        """ solve Ax=b for x

        Args:
            A (scipy.sparse.csr.csr_matrix): sparse square CSR matrix , CSC matrix also possible
            b (numpy.ndarray): right-hand side(s), b.shape[0] needs to be the same as A.shape[0]

        Returns:
            Solution of the system of linear equations, same shape as input b
        """
        return self._pardiso_solver.solve(self.A, b)

    def adjoint(self, b):
        # Cannot use _pardiso_solver.solve because it changes flag 12 internally
        iparm_prev = self._pardiso_solver.get_iparm(12)
        self._pardiso_solver.set_iparm(12, int(not iparm_prev))  # Adjoint solver (transpose)
        b = self._pardiso_solver._check_b(self.A, b)
        x = self._pardiso_solver._call_pardiso(self.A, b)
        self._pardiso_solver.set_iparm(12, iparm_prev)  # Revert back to normal solver
        return x

    def _print_iparm(self):
        """ Print all iparm settings to console """
        keys = ['Undefined' for _ in self._pardiso_solver.iparm]
        keys[0] = 'Use default values'
        keys[1] = 'Fill-in reducing ordering for the input matrix'
        keys[3] = 'Preconditioned CGS/CG'
        keys[4] = 'User permutation'
        keys[5] = 'Write solution on x'
        keys[6] = 'Number of iterative refinement steps performed'
        keys[7] = 'Iterative refinement step'
        keys[8] = 'Tolerance level for the relative residual in the iterative refinement process'
        keys[9] = 'Pivoting perturbation'
        keys[10] = 'Scaling vectors'
        keys[11] = 'Solve with transposed or conjugate transposed matrix A'
        keys[12] = 'Improved accuracy using (non-) symmetric weighted matching'
        keys[13] = 'Number of perturbed pivots'
        keys[14] = 'Peak memory on symbolic factorization'
        keys[15] = 'Permanent memory on symbolic factorization'
        keys[16] = 'Size of factors/Peak memory on numerical factorization and solution'
        keys[17] = 'Report the number of non-zero elements in the factors'
        keys[18] = 'Report number of floating point operations (in 10^6 floating point operations) that are necessary ' \
                   'to factor the matrix A'
        keys[19] = 'Report CG/CGS diagnostics'
        keys[20] = 'Pivoting for symmetric indefinite matrices'
        keys[21] = 'Inertia: number of positive eigenvalues'
        keys[22] = 'Inertia: number of negative eigenvalues'
        keys[23] = 'Parallel factorization control'
        keys[24] = 'Parallel forward/backward solve control'
        keys[26] = 'Matrix checker'
        keys[27] = 'Single or double precision Intel® oneAPI Math Kernel Library PARDISO'
        keys[29] = 'Number of zero or negative pivots'
        keys[30] = 'Partial solve and computing selected components of the solution vectors'
        keys[33] = 'Optimal number of OpenMP threads for conditional numerical reproducibility (CNR) mode'
        keys[34] = 'One- or zero-based indexing of columns and rows'
        keys[35] = 'Schur complement matrix computation control'
        keys[36] = 'Format for matrix storage'
        keys[37] = 'Enable low rank update to accelerate factorization for multiple matrices with identical structure ' \
                   'and similar values'
        keys[42] = 'Control parameter for the computation of the diagonal of inverse matrix'
        keys[55] = 'Diagonal and pivoting control'
        keys[59] = 'Intel® oneAPI Math Kernel Library PARDISO mode'
        keys[62] = 'Size of the minimum OOC memory for numerical factorization and solution'
        for i, (v, k) in enumerate(zip(self._pardiso_solver.iparm, keys)):
            if v != 0:
                print(f"{i+1}: {v} ({k})")  # i+1 because of 1-based numbering


# ------------------------------------ LU Solver -----------------------------------
try:
    from scikits.umfpack import splu  # UMFPACK solver; this one is faster and has identical interface
except ImportError:
    from scipy.sparse.linalg import splu


class SolverSparseLU(LinearSolver):
    """ Solver for sparse (square) matrices using an LU decomposition.

    Internally, `scipy` uses the SuperLU library, which is relatively slow. It may be sped up using the Python package
    ``scikit-umfpack``, which scipy` is able to use, but must be installed by the user.

    References:
      - `Scipy LU <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html>`_
      - `Scipy UMFPACK <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.use_solver.html>`_
    """
    def update(self, A):
        r"""  Factorize the matrix as :math:`\mathbf{A}=\mathbf{L}\mathbf{U}`, where :math:`\mathbf{L}` is a lower
        triangular matrix and :math:`\mathbf{U}` is upper triangular.
        """
        self.iscomplex = matrix_is_complex(A)
        self.inv = splu(A)
        return self

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{U}^{-1}\mathbf{L}^{-1}\mathbf{b}`.
        """
        return self.inv.solve(rhs)

    def adjoint(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A}^\text{H}\mathbf{x} = \mathbf{b}` by forward and
        backward substitution of :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{U}^{-\text{H}}\mathbf{b}`.
        """
        return self.inv.solve(rhs, trans=('H' if self.iscomplex else 'T'))


# ------------------------------------ Cholesky Solver scikit-sparse -----------------------------------
try:  # SCIKIT CHOLMOD
    """ Installation on Ubuntu22 using pip... Maybe it is also available in the conda repository
    sudo apt update
    sudo apt install libsuitesparse-dev
    pip install scikit-sparse
    """
    from sksparse import cholmod  # Sparse cholesky solver (CHOLMOD)
    _has_sksparse_cholmod = True
except ImportError:
    _has_sksparse_cholmod = False


class SolverSparseCholeskyScikit(LinearSolver):
    """ Solver for positive-definite Hermitian matrices using a Cholesky factorization.

    This solver requires the Python package ``scikit-sparse``. This package depends on the library ``suitesparse``,
    which can be more difficult to install on some systems. In case ``suitesparse`` cannot be installed,
    :class:`.SolverSparseCholeskyCVXOPT` is recommended, as installation of CVXOpt is easier and is packaged with
    ``suitesparse``.

    References:
      - `Scikit Installation <https://scikit-sparse.readthedocs.io/en/latest/overview.html>`_
      - `Scikit Cholmod <https://scikit-sparse.readthedocs.io/en/latest/cholmod.html>`_
    """

    defined = _has_sksparse_cholmod

    def __init__(self, A=None):
        super().__init__(A)
        if not self.defined:
            raise ImportError("scikit-sparse is not installed on this system")

    def update(self, A):
        r""" Factorize the matrix using Cholmod. In case the matrix :math:`\mathbf{A}` is non-Hermitian, the
        system of equations is solved in a least-squares sense:
        :math:`\min \left| \mathbf{A}\mathbf{x} - \mathbf{b} \right|^2`.
        The solution of this minimization is
        :math:`\mathbf{x}=(\mathbf{A}^\text{H}\mathbf{A})^{-1}\mathbf{A}^\text{H}\mathbf{b}`.
        """
        self.A = A
        if not hasattr(self, 'inv'):
            self.inv = cholmod.analyze(A)

        # Do the Cholesky factorization
        self.inv.cholesky_inplace(A)

        return self

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{L}^{-1}\mathbf{b}` in case of an
        Hermitian matrix.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.

        """
        return self.inv(rhs)

    def adjoint(self, rhs):
        return self.solve(rhs)


# ------------------------------------ Cholesky Solver cvxopt -----------------------------------
try:  # CVXOPT cholmod
    import cvxopt
    import cvxopt.cholmod
    _has_cvxopt_cholmod = True
except ImportError:
    _has_cvxopt_cholmod = False


class SolverSparseCholeskyCVXOPT(LinearSolver):
    """ Solver for positive-definite Hermitian matrices using a Cholesky factorization.

    This solver requires the Python package ``cvxopt``.

    References:
      - `CVXOPT Installation <http://cvxopt.org/install/index.html>`_
      - `CVXOPT Cholmod <https://cvxopt.org/userguide/spsolvers.html#positive-definite-linear-equations>`_
    """

    defined = _has_cvxopt_cholmod

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.defined:
            raise ImportError("cvxopt is not installed on this system")
        self._dtype = None
        self.inv = None

    def update(self, A):
        r""" Factorize the matrix using CVXOPT's Cholmod as :math:`\mathbf{A}=\mathbf{L}\mathbf{L}^\text{H}`. """
        if not isinstance(A, cvxopt.spmatrix):
            if not isinstance(A, sps.coo_matrix):
                Kcoo = A.tocoo()
                warnings.warn(f"{type(self).__name__}: Efficiency warning: CVXOPT spmatrix must be used")
            else:
                Kcoo = A
            K = cvxopt.spmatrix(Kcoo.data, Kcoo.row.astype(int), Kcoo.col.astype(int))
        else:
            K = A

        if self.inv is None:
            self.inv = cvxopt.cholmod.symbolic(K)
        cvxopt.cholmod.numeric(K, self.inv)
        if self._dtype is None:
            self._dtype = A.dtype

        return self

    def solve(self, rhs):
        r""" Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{L}^{-1}\mathbf{b}`. """
        if rhs.dtype != self._dtype:
            warnings.warn(f"{type(self).__name__}: Type warning: rhs value type ({rhs.dtype}) is converted to {self._dtype}")
        B = cvxopt.matrix(rhs.astype(self._dtype))
        nrhs = 1 if rhs.ndim == 1 else rhs.shape[1]
        cvxopt.cholmod.solve(self.inv, B, nrhs=nrhs)
        return np.array(B).flatten() if nrhs == 1 else np.array(B)

    def adjoint(self, rhs):
        return self.solve(rhs)
