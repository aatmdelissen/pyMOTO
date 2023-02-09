import os
import sys
import glob
import ctypes
import hashlib
import warnings
from ctypes.util import find_library  # To find MKL
import numpy as np
import scipy.sparse as sps
from scipy.sparse import SparseEfficiencyWarning
from .solvers import matrix_is_hermitian, matrix_is_complex, matrix_is_symmetric, LinearSolver


def _find_libmkl():
    libmkl = None
    # Look for the mkl_rt shared library with ctypes.util.find_library
    mkl_rt = find_library('mkl_rt')
    # also look for mkl_rt.1, Windows-specific, see
    # https://github.com/haasad/PyPardisoProject/issues/12
    if mkl_rt is None:
        mkl_rt = find_library('mkl_rt.1')

    # If we can't find mkl_rt with find_library, we search the directory
    # tree, using a few assumptions:
    # - the shared library can be found in a subdirectory of sys.prefix
    #   https://docs.python.org/3.9/library/sys.html#sys.prefix
    # - either in `lib` (linux and macOS) or `Library\bin` (windows)
    # - if there are multiple matches for `mkl_rt`, try shorter paths
    #   first
    if mkl_rt is None:
        roots = [sys.prefix]
        try:
            roots.append(os.environ['MKLROOT'])
        except KeyError:
            pass

        for root in roots:
            mkl_rt_path = sorted(
                glob.glob(f'{root}/[Ll]ib*/**/*mkl_rt*', recursive=True),
                key=len
            )
            for path in mkl_rt_path:
                try:
                    libmkl = ctypes.CDLL(path)
                    break
                except (OSError, ImportError):
                    pass
            if libmkl is not None:
                break

        if libmkl is None:
            raise ImportError('Shared library mkl_rt not found')
    else:
        libmkl = ctypes.CDLL(mkl_rt)
    return libmkl


try:
    __libmkl__ = _find_libmkl()
except ImportError:
    __libmkl__ = None


class SolverSparsePardiso(LinearSolver):
    """ Solver wrapper Intel MKL Pardiso solver, which is a very fast and flexible multi-threaded solver

    Forked from https://github.com/haasad/PyPardisoProject

    TODO: This project has been active again, so it would be better to use this package directly

    Python interface to the Intel MKL PARDISO library for solving large sparse linear systems of equations Ax=b.
    Pardiso documentation: https://software.intel.com/en-us/node/470282

    Basic usage
        matrix type: real (float64) and nonsymetric

        methods: ``solve``, ``factorize``

          - use the "solve(A,b)" method to solve Ax=b for x, where A is a sparse CSR (or CSC) matrix and b is a numpy array
          - use the "factorize(A)" method first, if you intend to solve the system more than once for different right-hand
            sides, the factorization will be reused automatically afterwards

    Advanced usage
        methods: ``get_iparm``, ``get_iparms``, ``set_iparm``, ``set_matrix_type``, ``set_phase``

          - additional options can be accessed by setting the iparms (see Pardiso documentation for description)
          - other matrix types can be chosen with the "set_matrix_type" method. complex matrix types are currently not
            supported. pypardiso is only teste for mtype=11 (real and nonsymetric)
          - the solving phases can be set with the "set_phase" method
          - The out-of-core (OOC) solver either fails or crashes my computer, be careful with iparm[60]

    Statistical info
        methods: ``set_statistical_info_on``, ``set_statistical_info_off``

          - the Pardiso solver writes statistical info to the C stdout if desired
          - if you use pypardiso from within a jupyter notebook you can turn the statistical info on and capture the output
            real-time by wrapping your call to "solve" with wurlitzer.sys_pipes() (https://github.com/minrk/wurlitzer,
            https://pypi.python.org/pypi/wurlitzer/)
          - wurlitzer dosen't work on windows, info appears in notebook server console window if used from jupyter notebook

    Memory usage
        methods: ``remove_stored_factorization``, ``free_memory``

          - ``remove_stored_factorization`` can be used to delete the wrapper's copy of matrix A
          - ``free_memory`` releases the internal memory of the solver

    Args:
        A (optional): The matrix
        symmetric (optional): If it is already known if the matrix is symmetric, you can provide it here
        hermitian (optional): If it is already known if the matrix is Hermitian, you can provide it here
        positive_definite (optional): If positive-definiteness is known, provide it here

    References:
        `Intel MKL Pardiso <https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface.html>`_
    """

    defined = __libmkl__ is not None

    def __init__(self, A=None, symmetric=None, hermitian=None, positive_definite=None, size_limit_storage=5e7):
        super().__init__(A)
        if not self.defined:
            raise ImportError("Intel MKL Pardiso solver cannot be found. ")

        global __libmkl__
        self.libmkl = __libmkl__
        self._mkl_pardiso = self.libmkl.pardiso

        # determine 32bit or 64bit architecture
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self._pt_type = (ctypes.c_int64, np.int64)
        else:
            self._pt_type = (ctypes.c_int32, np.int32)

        self._mkl_pardiso.argtypes = [ctypes.POINTER(self._pt_type[0]),    # pt
                                      ctypes.POINTER(ctypes.c_int32),      # maxfct
                                      ctypes.POINTER(ctypes.c_int32),      # mnum
                                      ctypes.POINTER(ctypes.c_int32),      # mtype
                                      ctypes.POINTER(ctypes.c_int32),      # phase
                                      ctypes.POINTER(ctypes.c_int32),      # n
                                      ctypes.POINTER(None),                # a
                                      ctypes.POINTER(ctypes.c_int32),      # ia
                                      ctypes.POINTER(ctypes.c_int32),      # ja
                                      ctypes.POINTER(ctypes.c_int32),      # perm
                                      ctypes.POINTER(ctypes.c_int32),      # nrhs
                                      ctypes.POINTER(ctypes.c_int32),      # iparm
                                      ctypes.POINTER(ctypes.c_int32),      # msglvl
                                      ctypes.POINTER(None),                # b
                                      ctypes.POINTER(None),                # x
                                      ctypes.POINTER(ctypes.c_int32)]      # error

        self._mkl_pardiso.restype = None

        self.pt = np.zeros(64, dtype=self._pt_type[1])
        self.iparm = np.zeros(64, dtype=np.int32)
        self.perm = np.zeros(0, dtype=np.int32)

        self.mtype = None
        self.hermitian = hermitian
        self.symmetric = symmetric
        self.positive_definite = positive_definite
        self.phase = 13
        self.msglvl = False  # Show debug info

        self.factorized_A = sps.csr_matrix((0, 0))
        self.size_limit_storage = size_limit_storage
        self._solve_transposed = False
        self._is_factorized = False

        self.set_iparm(0, 1)  # Use non-default values
        self.set_iparm(34, 1)  # Zero-based indexing, as is done in Python

    def determine_mtype(self, A):
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
            `Intel MKL Pardiso arguments <https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso.html#pardiso>`_

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
        if self.mtype is None:
            self.mtype = self.determine_mtype(A)

        if self.mtype in {-2, 2, 6}:
            A = sps.triu(A, format='csr')  # Only use upper part
        if not sps.isspmatrix_csr(A):
            warnings.warn(f'PyPardiso requires CSR matrix format, not {type(A).__name__}', SparseEfficiencyWarning)
            A = A.tocsr()
        self.A = A
        self._check_A(A)

        if A.nnz > self.size_limit_storage:
            self.factorized_A = self._hash_csr_matrix(A)
        else:
            self.factorized_A = A.copy()

        self.set_phase(12)
        b = np.zeros((A.shape[0], 1), dtype=self.dtype)
        self._call_pardiso(A, b)
        self._is_factorized = True

    def solve(self, b):
        """ solve Ax=b for x

        Args:
            A (scipy.sparse.csr.csr_matrix): sparse square CSR matrix , CSC matrix also possible
            b (numpy.ndarray): right-hand side(s), b.shape[0] needs to be the same as A.shape[0]

        Returns:
            Solution of the system of linear equations, same shape as input b
        """
        # self._check_A(A)  # TODO
        b = self._check_b(self.A, b)

        if self._is_factorized:
            self.set_phase(33)
        else:
            self.set_phase(13)

        x = self._call_pardiso(self.A, b)
        return x

    def adjoint(self, b):
        b = self._check_b(self.A, b)
        self.set_iparm(11, 1)  # Adjoint solver
        x = self._call_pardiso(self.A, b)
        self.set_iparm(11, 0)  # Revert back to normal solver
        return x

    def _is_already_factorized(self, A):
        """ Check if the matrix is already factorized """
        if type(self.factorized_A) == str:
            return self._hash_csr_matrix(A) == self.factorized_A
        else:
            return self._csr_matrix_equal(A, self.factorized_A)

    def _csr_matrix_equal(self, a1, a2):
        return all((np.array_equal(a1.indptr, a2.indptr),
                    np.array_equal(a1.indices, a2.indices),
                    np.array_equal(a1.data, a2.data)))

    def _hash_csr_matrix(self, matrix):
        return (hashlib.sha1(matrix.indices).hexdigest() +
                hashlib.sha1(matrix.indptr).hexdigest() +
                hashlib.sha1(matrix.data).hexdigest())

    def _check_A(self, A):
        if A.shape[0] != A.shape[1]:
            raise ValueError('Matrix A needs to be square, but has shape: {}'.format(A.shape))

        if sps.isspmatrix_csr(A):
            self._solve_transposed = False
            self.set_iparm(11, 0)
        else:
            raise ValueError(f'PyPardiso requires CSR matrix format, not {type(A).__name__}')

        # scipy allows unsorted csr-indices, which lead to completely wrong pardiso results
        if not A.has_sorted_indices:
            A.sort_indices()

        # scipy allows csr matrices with empty rows. a square matrix with an empty row is singular. calling
        # pardiso with a matrix A that contains empty rows leads to a segfault, same applies for csc with
        # empty columns
        # if not np.diff(A.indptr).all():
        #     row_col = 'column' if self._solve_transposed else 'row'
        #     raise ValueError('Matrix A is singular, because it contains empty {}(s)'.format(row_col))

        if A.dtype != self.dtype:
            raise TypeError(f'PyPardiso only supports float64 and complex128, but matrix A has dtype: {A.dtype}')

    def _check_b(self, A, b):
        if sps.isspmatrix(b):
            warnings.warn('PyPardiso requires the right-hand side b to be a dense array for maximum efficiency',
                          SparseEfficiencyWarning)
            b = b.todense()

        # pardiso expects fortran (column-major) order if b is a matrix
        if b.ndim == 2:
            b = np.asfortranarray(b)

        if b.shape[0] != A.shape[0]:
            raise ValueError("Dimension mismatch: Matrix A {} and array b {}".format(A.shape, b.shape))

        if b.dtype != self.dtype:
            warnings.warn(f"Array b's data type is converted from {b.dtype} to {self.dtype}", PyPardisoWarning)
            b = b.astype(self.dtype)

        return b

    def _call_pardiso(self, A, b):
        x = np.zeros_like(b, dtype=self.dtype)
        pardiso_error = ctypes.c_int32(0)
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        c_float64_p = ctypes.POINTER(ctypes.c_double)

        # https://stackoverflow.com/questions/13373291/complex-number-in-ctypes
        class c_double_complex(ctypes.Structure):
            """complex is a c structure
            https://docs.python.org/3/library/ctypes.html#module-ctypes suggests
            to use ctypes.Structure to pass structures (and, therefore, complex)
            """
            _fields_ = [("real", ctypes.c_double),("imag", ctypes.c_double)]
            @property
            def value(self):
                return self.real+1j*self.imag # fields declared above
        c_complex128_p = ctypes.POINTER(c_double_complex)

        dat_t = c_complex128_p if self.complex else c_float64_p

        # 1-based indexing
        ia = A.indptr
        ja = A.indices
        # self.print_iparm()
        self._mkl_pardiso(self.pt.ctypes.data_as(ctypes.POINTER(self._pt_type[0])),  # pt
                          ctypes.byref(ctypes.c_int32(1)),  # maxfct
                          ctypes.byref(ctypes.c_int32(1)),  # mnum
                          ctypes.byref(ctypes.c_int32(self.mtype)),  # mtype -> 11 for real-nonsymetric
                          ctypes.byref(ctypes.c_int32(self.phase)),  # phase -> 13
                          ctypes.byref(ctypes.c_int32(A.shape[0])),  # N -> number of equations/size of matrix
                          A.data.ctypes.data_as(dat_t),  # A -> non-zero entries in matrix
                          ia.ctypes.data_as(c_int32_p),  # ia -> csr-indptr
                          ja.ctypes.data_as(c_int32_p),  # ja -> csr-indices
                          self.perm.ctypes.data_as(c_int32_p),  # perm -> empty
                          ctypes.byref(ctypes.c_int32(1 if b.ndim == 1 else b.shape[1])),  # nrhs
                          self.iparm.ctypes.data_as(c_int32_p),  # iparm-array
                          ctypes.byref(ctypes.c_int32(self.msglvl)),  # msg-level -> 1: statistical info is printed
                          b.ctypes.data_as(dat_t),  # b -> right-hand side vector/matrix
                          x.ctypes.data_as(dat_t),  # x -> output
                          ctypes.byref(pardiso_error))  # pardiso error

        if pardiso_error.value != 0:
            raise PyPardisoError(pardiso_error.value)
        else:
            return np.ascontiguousarray(x)  # change memory-layout back from fortran to c order

    def get_iparms(self):
        """ Returns a dictionary of iparms """
        return dict(enumerate(self.iparm, 1))

    def get_iparm(self, i):
        """ Returns the i-th iparm (1-based indexing) """
        return self.iparm[i]

    def set_iparm(self, i, value):
        """ set the i-th iparm to 'value' (1-based indexing)

        References:
            `Intel documentation <https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html>`_
        """
        if i not in {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 2426, 27, 29, 30, 33, 34, 35, 36, 38, 42, 55, 59, 62}:
            warnings.warn('{} is no input iparm. See the Pardiso documentation.'.format(value), PyPardisoWarning)
        self.iparm[i] = value

    def print_iparm(self):
        """ Print all iparm settings to console """
        keys = ['Undefined' for _ in self.iparm]
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
        keys[18] = 'Report number of floating point operations (in 10^6 floating point operations) that are necessary to factor the matrix A'
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
        keys[37] = 'Enable low rank update to accelerate factorization for multiple matrices with identical structure and similar values'
        keys[42] = 'Control parameter for the computation of the diagonal of inverse matrix'
        keys[55] = 'Diagonal and pivoting control'
        keys[59] = 'Intel® oneAPI Math Kernel Library PARDISO mode'
        keys[62] = 'Size of the minimum OOC memory for numerical factorization and solution'
        for i, (v, k) in enumerate(zip(self.iparm, keys)):
            if v != 0:
                print(f"{i}: {v} ({k})")

    def set_matrix_type(self, mtype):
        """ Set the matrix type (see Pardiso documentation) """
        self.mtype = mtype

    def set_statistical_info_on(self):
        """ Display statistical info (appears in notebook server console window if pypardiso is
        used from jupyter notebook, use wurlitzer to redirect info to the notebook) """
        self.msglvl = 1

    def set_statistical_info_off(self):
        """ Turns statistical info off """
        self.msglvl = 0

    def set_phase(self, phase):
        """ Set the phase(s) for the solver. See the Pardiso documentation for details.

        phase: Solver Execution Steps
          - ``11`` Analysis
          - ``12`` Analysis, numerical factorization
          - ``13`` Analysis, numerical factorization, solve, iterative refinement
          - ``22`` Numerical factorization
          - ``23`` Numerical factorization, solve, iterative refinement
          - ``33`` Solve, iterative refinement
          - ``331`` like phase=33, but only forward substitution
          - ``332`` like phase=33, but only diagonal substitution (if available)
          - ``333`` like phase=33, but only backward substitution
          - ``0`` Release internal memory for L and U matrix number mnum
          - ``-1`` Release all internal memory for all matrices
        """
        self.phase = phase

    def remove_stored_factorization(self):
        """ removes the stored factorization, this will free the memory in python, but the factorization in pardiso
        is still accessible with a direct call to self._call_pardiso(A,b) with phase=33 """
        self.factorized_A = sps.csr_matrix((0, 0))

    def free_memory(self, everything=False):
        """ release mkl's internal memory, either only for the factorization (ie the LU-decomposition) or all of
        mkl's internal memory if everything=True """
        self.remove_stored_factorization()
        A = sps.csr_matrix((0, 0))
        b = np.zeros(0)
        self.set_phase(-1 if everything else 0)
        self._call_pardiso(A, b)
        self.set_phase(13)

    def __del__(self):
        self.free_memory(everything=True)


class PyPardisoWarning(UserWarning):
    pass


class PyPardisoError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        error_msg = dict()
        error_msg[0] = "No error"
        error_msg[-1] = "Input inconsistent"
        error_msg[-2] = "Not enough memory"
        error_msg[-3] = "Reordering problem"
        error_msg[-4] = "Zero pivot"
        error_msg[-5] = "Unclassified (internal) error"
        error_msg[-6] = "Reordering failed"
        error_msg[-7] = "Diagonal matrix is singular"
        error_msg[-8] = "32-bit integer overflow problem"
        error_msg[-9] = "Not enough memory for OOC"
        error_msg[-10] = "Error opening OOC files"
        error_msg[-11] = "Read/write error with OOC files"
        error_msg[-12] = "pardiso_64 called from 32-bit library"
        error_msg[-13] = "Interrupted by the mkl_progress function"
        error_msg[-15] = "Internal error for iparm[23]=10 and iparm[12]=1"

        try:
            info = error_msg[self.value]
        except KeyError:
            info = "Unknown"

        return (f'The Pardiso solver failed with error code {self.value} : {info}. '
                'See Pardiso documentation for details.')


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
        """  Factorize the matrix as :math:`\mathbf{A}=\mathbf{L}\mathbf{U}`, where :math:`\mathbf{L}` is a lower
        triangular matrix and :math:`\mathbf{U}` is upper triangular.
        """
        self.iscomplex = matrix_is_complex(A)
        self.inv = splu(A)
        return self

    def solve(self, rhs):
        """ Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{U}^{-1}\mathbf{L}^{-1}\mathbf{b}`.
        """
        return self.inv.solve(rhs)

    def adjoint(self, rhs):
        """ Solves the linear system of equations :math:`\mathbf{A}^\\text{H}\mathbf{x} = \mathbf{b}` by forward and
        backward substitution of :math:`\mathbf{x} = \mathbf{L}^{-\\text{H}}\mathbf{U}^{-\\text{H}}\mathbf{b}`.
        """
        return self.inv.solve(rhs, trans=('H' if self.iscomplex else 'T'))


# Sparse cholesky solver
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
            raise ImportError(f"scikit-sparse is not installed on this system")

    def update(self, A):
        """ Factorize the matrix using Cholmod. In case the matrix :math:`\mathbf{A}` is non-Hermitian, the
        system of equations is solved in a least-squares sense:
        :math:`\min \left| \mathbf{A}\mathbf{x} - \mathbf{b} \\right|^2`.
        The solution of this minimization is
        :math:`\mathbf{x}=(\mathbf{A}^\\text{H}\mathbf{A})^{-1}\mathbf{A}^\\text{H}\mathbf{b}`.
        """
        self.A = A
        if not hasattr(self, 'inv'):
            self.inv = cholmod.analyze(A)

        # Do the Cholesky factorization
        self.inv.cholesky_inplace(A)

        return self

    def solve(self, rhs):
        """ Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{L}^{-\\text{H}}\mathbf{L}^{-1}\mathbf{b}` in case of an
        Hermitian matrix.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.

        """
        return self.inv(rhs)

    def adjoint(self, rhs):
        return self.solve(rhs)


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
            raise ImportError(f"cvxopt is not installed on this system")
        self._dtype = None
        self.inv = None

    def update(self, A):
        """ Factorize the matrix using CVXOPT's Cholmod as :math:`\mathbf{A}=\mathbf{L}\mathbf{L}^\\text{H}`. """
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
        """ Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{L}^{-\\text{H}}\mathbf{L}^{-1}\mathbf{b}`. """
        if rhs.dtype != self._dtype:
            warnings.warn(f"{type(self).__name__}: Type warning: rhs value type ({rhs.dtype}) is converted to {self._dtype}")
        B = cvxopt.matrix(rhs.astype(self._dtype))
        nrhs = 1 if rhs.ndim == 1 else rhs.shape[1]
        cvxopt.cholmod.solve(self.inv, B, nrhs=nrhs)
        return np.array(B).flatten() if nrhs == 1 else np.array(B)

    def adjoint(self, rhs):
        return self.solve(rhs)
