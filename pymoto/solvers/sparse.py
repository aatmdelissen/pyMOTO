import sys
import glob
import ctypes
import warnings
import hashlib
from ctypes.util import find_library

import numpy as np
import scipy.sparse as sps
from scipy.sparse import SparseEfficiencyWarning
from .matrix_checks import matrix_is_hermitian, matrix_is_complex, matrix_is_symmetric
from .solvers import LinearSolver


# ------------------------------------ Pardiso Solver -----------------------------------
def load_libmkl():
    # Code borrowed from: https://github.com/andrejstmh/PyPardisoProject/blob/master/pypardiso/pardiso_wrapper.py
    libmkl = None
    # Look for the mkl_rt shared library with ctypes.util.find_library
    mkl_rt = find_library("mkl_rt")
    # also look for mkl_rt.1, Windows-specific, see
    # https://github.com/haasad/PyPardisoProject/issues/12
    if mkl_rt is None:
        mkl_rt = find_library("mkl_rt.1")

    # If we can't find mkl_rt with find_library, we search the directory
    # tree, using a few assumptions:
    # - the shared library can be found in a subdirectory of sys.prefix
    #   https://docs.python.org/3.9/library/sys.html#sys.prefix
    # - either in `lib` (linux and macOS) or `Library\bin` (windows)
    # - if there are multiple matches for `mkl_rt`, try shorter paths
    #   first
    if mkl_rt is None:
        mkl_rt_path = sorted(glob.glob(f"{sys.prefix}/[Ll]ib*/**/*mkl_rt*", recursive=True), key=len)
        for path in mkl_rt_path:
            try:
                libmkl = ctypes.CDLL(path)
                break
            except (OSError, ImportError):
                pass
        if libmkl is None:
            raise ImportError("Shared library mkl_rt not found")
    else:
        libmkl = ctypes.CDLL(mkl_rt)
    return libmkl


try:
    libmkl = load_libmkl()
except ImportError:
    libmkl = None


class IparmOptions:
    """https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pardiso-iparm-parameter.html"""

    keys = [None for _ in range(64)]
    keys[0] = "Use default values"
    keys[1] = "Fill-in reducing ordering for the input matrix"
    keys[3] = "Preconditioned CGS/CG"
    keys[4] = "User permutation"
    keys[5] = "Write solution on x"
    keys[6] = "Number of iterative refinement steps performed"
    keys[7] = "Iterative refinement step"
    keys[8] = "Tolerance level for the relative residual in the iterative refinement process"
    keys[9] = "Pivoting perturbation"
    keys[10] = "Scaling vectors"
    keys[11] = "Solve with transposed or conjugate transposed matrix A"
    keys[12] = "Improved accuracy using (non-) symmetric weighted matching"
    keys[13] = "Number of perturbed pivots"
    keys[14] = "Peak memory on symbolic factorization"
    keys[15] = "Permanent memory on symbolic factorization"
    keys[16] = "Size of factors/Peak memory on numerical factorization and solution"
    keys[17] = "Report the number of non-zero elements in the factors"
    keys[18] = "Report number of floating point operations (in 10^6 FLOPS) that are necessary to factor the matrix A"
    keys[19] = "Report CG/CGS diagnostics"
    keys[20] = "Pivoting for symmetric indefinite matrices"
    keys[21] = "Inertia: number of positive eigenvalues"
    keys[22] = "Inertia: number of negative eigenvalues"
    keys[23] = "Parallel factorization control"
    keys[24] = "Parallel forward/backward solve control"
    keys[26] = "Matrix checker"
    keys[27] = "Single or double precision (1=single; 0=double)"
    keys[29] = "Number of zero or negative pivots"
    keys[30] = "Partial solve and computing selected components of the solution vectors"
    keys[33] = "Optimal number of OpenMP threads for conditional numerical reproducibility (CNR) mode"
    keys[34] = "One- or zero-based indexing of columns and rows (1=zero-based; 0=one-based)"
    keys[35] = "Schur complement matrix computation control"
    keys[36] = "Format for matrix storage"
    keys[37] = """Enable low rank update to accelerate factorization for multiple matrices with identical structure and 
    similar values"""
    keys[42] = "Control parameter for the computation of the diagonal of inverse matrix"
    keys[55] = "Diagonal and pivoting control"
    keys[59] = "Intel® oneAPI Math Kernel Library PARDISO mode"
    keys[62] = "Size of the minimum OOC memory for numerical factorization and solution"

    def __init__(self, mtype=None):
        self.data = np.zeros(64, dtype=np.int32)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, value):
        if self.keys[i] is None:
            warnings.warn(f"#{i} is no input iparm. See Pardiso documentation.")
        self.data[i] = value

        # description = self.keys[i] if self.keys[i] is not None else "Undefined"
        # print(f"iparm[{i}] = {self.data[i]}  # {description}")

    def print_all(self):
        """Print all iparm settings to console"""
        for i in range(self.data.size):
            if self.keys[i] is not None or self.data[i] != 0:
                description = self.keys[i] if self.keys[i] is not None else "Undefined"
                print(f"iparm[{i}] = {self.data[i]}  # {description}")


class PardisoWarning(UserWarning):
    pass


class PardisoError(Exception):
    _error_codes = dict()
    _error_codes[-1] = "Input inconsistent"
    _error_codes[-2] = "Not enough memory"
    _error_codes[-3] = "Not enough memory"
    _error_codes[-4] = "Zero pivot, numerical factorization or iterative refinement problem"
    _error_codes[-5] = "Unclassifed (internal) error"
    _error_codes[-6] = "Reordering failed (`mtype` 11 and 13 only)"
    _error_codes[-7] = "Diagonal matrix is singular"
    _error_codes[-8] = "32-bit integer overflow problem"
    _error_codes[-9] = "Not enough memory for OOC"
    _error_codes[-10] = "Error opening OOC files"
    _error_codes[-11] = "Read/write error with OOC files"
    _error_codes[-12] = "Pardiso_64 called from 32-bit library"
    _error_codes[-13] = "Interrupted by the mkl_progress function"
    _error_codes[-15] = "Internal error which can appear for `iparm[23]=10` and `iparm[12]=1`"
    
    def __init__(self, value):
        self.value = value

    def __str__(self):
        err = self._error_codes[self.value]
        return f"The Pardiso solver failed with error code {self.value} ({err}). See Pardiso documentation for details."


class SolverSparsePardiso(LinearSolver):
    r"""Solver wrapper Intel MKL Pardiso solver, which is a very fast and flexible multi-threaded solver

    Complex-valued matrices are currently not supported

    Uses the Python interface to the Intel MKL PARDISO library for solving large sparse linear systems of
    equations :math:`\mathbf{Ax}=\mathbf{b}`.

    References:
        `Intel MKL Python wrapper <https://pypi.org/project/mkl/>`_
        `Intel MKL Pardiso <https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/
          top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface.html>`_
    """

    defined = libmkl is not None
    _supported_types = [np.single, np.csingle, np.double, np.cdouble]

    def __init__(
        self,
        A: sps.spmatrix = None,
        symmetric: bool = None,
        hermitian: bool = None,
        positive_definite: bool = None,
        size_limit_storage: int = 5e7,
    ):
        """Initialize Pardiso linear solver

        Args:
            A (scipy.sparse.spmatrix, optional): The matrix. Defaults to None.
            symmetric (bool, optional): If it is already known if the matrix is symmetric, you can provide it here
            hermitian (bool, optional): If it is already known if the matrix is Hermitian, you can provide it here
            positive_definite (bool, optional): If positive-definiteness is known, provide it here
            size_limit_storage (int, optional): Limit for MKL memory use
        """

        if not self.defined:
            raise ImportError("Intel MKL Pardiso solver (mkl) cannot be found. ")

        self.hermitian = hermitian
        self.symmetric = symmetric
        self.positive_definite = positive_definite

        self._mkl_pardiso = libmkl.pardiso

        # determine 32bit or 64bit architecture
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self._pt_type = (ctypes.c_int64, np.int64)
        else:
            self._pt_type = (ctypes.c_int32, np.int32)

        self._mkl_pardiso.argtypes = [
            ctypes.POINTER(self._pt_type[0]),  # pt
            ctypes.POINTER(ctypes.c_int32),  # maxfct
            ctypes.POINTER(ctypes.c_int32),  # mnum
            ctypes.POINTER(ctypes.c_int32),  # mtype
            ctypes.POINTER(ctypes.c_int32),  # phase
            ctypes.POINTER(ctypes.c_int32),  # n
            ctypes.POINTER(None),  # a (void*)
            ctypes.POINTER(ctypes.c_int32),  # ia
            ctypes.POINTER(ctypes.c_int32),  # ja
            ctypes.POINTER(ctypes.c_int32),  # perm
            ctypes.POINTER(ctypes.c_int32),  # nrhs
            ctypes.POINTER(ctypes.c_int32),  # iparm
            ctypes.POINTER(ctypes.c_int32),  # msglvl
            ctypes.POINTER(None),  # b (void*)
            ctypes.POINTER(None),  # x (void*)
            ctypes.POINTER(ctypes.c_int32),
        ]  # error

        self._mkl_pardiso.restype = None

        self._pt = np.zeros(64, dtype=self._pt_type[1])
        self._iparm = IparmOptions()
        self._perm = np.zeros(0, dtype=np.int32)
        self._dtype = None  # Dtype of matrix, rhs and solution should be the same

        self._mtype = None
        """ Phases: 1) is analysis and symbolic factorization, 2) is numerical factorization, 3) fw/bw substitution and 
            iterative refinement
          - `11` Analysis
          - `12` Analysis, numerical factorization
          - `13` Analysis, numerical factorization, solve, iterative refinement
          - `22` Numerical factorization
          - `23` Numerical factorication, solve, iterative refinement
          - `33` Solve, iterative refinement 
          - `331` Like `33`, but only forward substitution
          - `332` Like `33`, but only diagonal substitution
          - `333` Like `33`, but only backward substitution
          - `0` Release internal memory for L and U matrix
          - `-1` Release all internal memory for all matrices
        """
        self._phase = None
        self._msglvl = 0  # 1 shows statistical info of MKL

        self._factorized_A = sps.csr_matrix((0, 0))
        self.size_limit_storage = size_limit_storage
        self._solve_transposed = False

        super().__init__(A)

    def _check_A(self, A):
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix A needs to be square, but has shape: {A.shape}")

        if sps.isspmatrix_csc(A):
            self._solve_transposed = True
        else:
            if not sps.isspmatrix_csr(A):
                msg = f"Pardiso requires csr or csc matrix. Converted from {type(A).__name__} to csr."
                warnings.warn(msg, SparseEfficiencyWarning)
                A = sps.csr_matrix(A)
            self._solve_transposed = False

        # scipy allows unsorted csr-indices, which lead to completely wrong pardiso results
        if not A.has_sorted_indices:
            A.sort_indices()

        # scipy allows csr matrices with empty rows. a square matrix with an empty row is singular. calling
        # pardiso with a matrix A that contains empty rows leads to a segfault, same applies for csc with
        # empty columns
        if not np.diff(A.indptr).all():
            row_col = "column" if self._solve_transposed else "row"
            raise ValueError(f"Matrix A is singular, because it contains empty {row_col}(s)")

        if not any([np.issubdtype(A.dtype, t) for t in self._supported_types]):
            raise TypeError("Pardiso only supports {self._supported_types}, but matrix A has dtype: {A.dtype}")

        self._mtype = self._determine_mtype(A)

        # https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-0/pardiso-iparm-parameter.html
        if self._iparm[0] == 0:
            # Set default parameters
            c_int32_p = ctypes.POINTER(ctypes.c_int32)
            libmkl.pardisoinit(
                self._pt.ctypes.data_as(ctypes.POINTER(self._pt_type[0])),  # pt
                ctypes.byref(ctypes.c_int32(self._mtype)),  # mtype
                self._iparm.data.ctypes.data_as(c_int32_p),
            )  # pardiso error

        # Overwrite defaults
        self._iparm[1] = 3  # Makes solving a bit faster (~10%) using parallel dissection algorithm (improves phase 1)
        if self._mtype in {1, 2, 3, -4, 4}:  # Symmetric matrix
            self._iparm[9] = 0  # Set pivoting perturbation to zero; default value doesn't work for these matrix types

        is_single_precision = np.issubdtype(A.dtype, np.single) or np.issubdtype(A.dtype, np.csingle)
        self._iparm[27] = is_single_precision
        # This option may make solving faster, but with benchmarks it didn't seem to do so
        # if self._mtype in {11, 13}:  # Non-symmetric matrix
        #     self._iparm[23] = 10  # use improved two-level factorization algorithm
        # else:
        #     self._iparm[23] = 1 # use parallel factorization algorithm. (does not work if iparm[10] and/or
        #                         # iparm[12] are not 0, i.e. if scaling and/or matching are enabled
        self._iparm[34] = 1  # Use 0-based indexing


        # For symmetric or hermitian matrices, only the upper triangular part is used.
        if self._mtype in {-2, 2, -4, 4, 6}:
            A = sps.triu(A, format="coo")
            # Explicitly set zero diagonal entries, as this is better for Intel Pardiso
            (zero_diag_entries,) = np.where(A.diagonal() == 0)
            if len(zero_diag_entries) > 0:
                A.row = np.append(A.row, zero_diag_entries)
                A.col = np.append(A.col, zero_diag_entries)
                A.data = np.append(A.data, np.zeros_like(zero_diag_entries))
            A = A.tocsr()
            self._solve_transposed = False  # CSR matrix again so cancel transposition
        return A

    def _check_b(self, b):
        if sps.isspmatrix(b):
            warnings.warn(
                "Pardiso requires the right-hand side b to be a dense array for maximum efficiency",
                SparseEfficiencyWarning,
            )
            b = b.todense()

        # pardiso expects fortran (column-major) order for b
        if not b.flags.f_contiguous:
            b = np.asfortranarray(b)

        if not np.issubdtype(b.dtype, self.A.dtype):
            if np.iscomplexobj(b) and not np.iscomplexobj(self.A):
                raise TypeError(f"Complex b ({b.dtype}) cannot be used with real matrix A ({self.A.dtype})")
            warnings.warn(f"Array b's data type was converted from {b.dtype} to {self.A.dtype}", PardisoWarning)
            b = b.astype(self.A.dtype)

        return b

    def _determine_mtype(self, A):
        """Determines the matrix-type according to Intel

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
        # TODO structural symmetry is not checked, may be added at some point (1 and 3). For now 11 and 13 are used.
        self.complex = matrix_is_complex(A)
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

    def _call_pardiso(self, A: sps.spmatrix = None, b: np.ndarray = None):
        if A is None:
            A = self.A
        if b is None:
            b = np.zeros((A.shape[0], 1), dtype=A.dtype, order='F')  # Dummy rhs
        # Convert rhs to be contiguous (F)
        # Sometimes slicing (e.g. in LDA wrapper) may introduce unwanted non-contiguousness
        b = np.require(b, requirements='F')
        x = np.zeros_like(b, dtype=A.dtype)
        pardiso_error = ctypes.c_int32(0)
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        if self._iparm[27]:  # single precision
            c_data_p = ctypes.POINTER(ctypes.c_float)
        else:
            c_data_p = ctypes.POINTER(ctypes.c_double)
                    
        self._mkl_pardiso(
            self._pt.ctypes.data_as(ctypes.POINTER(self._pt_type[0])),  # pt
            ctypes.byref(ctypes.c_int32(1)),  # maxfct
            ctypes.byref(ctypes.c_int32(1)),  # mnum
            ctypes.byref(ctypes.c_int32(self._mtype)),  # mtype
            ctypes.byref(ctypes.c_int32(self._phase)),  # phase
            ctypes.byref(ctypes.c_int32(A.shape[0])),  # N -> number of equations/size of matrix
            A.data.ctypes.data_as(c_data_p),  # A -> non-zero entries in matrix
            A.indptr.ctypes.data_as(c_int32_p),  # ia -> csr-indptr
            A.indices.ctypes.data_as(c_int32_p),  # ja -> csr-indices
            self._perm.ctypes.data_as(c_int32_p),  # perm -> empty
            ctypes.byref(ctypes.c_int32(1 if b.ndim == 1 else b.shape[1])),  # nrhs
            self._iparm.data.ctypes.data_as(c_int32_p),  # iparm-array
            ctypes.byref(ctypes.c_int32(self._msglvl)),  # msg-level -> 1: statistical info is printed
            b.ctypes.data_as(c_data_p),  # b -> right-hand side vector/matrix
            x.ctypes.data_as(c_data_p),  # x -> output
            ctypes.byref(pardiso_error),
        )  # pardiso error

        if pardiso_error.value != 0:
            raise PardisoError(pardiso_error.value)
        else:
            return np.ascontiguousarray(x)  # change memory-layout back from fortran to c order

    @staticmethod
    def _hash_csr_matrix(self, matrix):
        return (
            hashlib.sha1(matrix.indices).hexdigest()
            + hashlib.sha1(matrix.indptr).hexdigest()
            + hashlib.sha1(matrix.data).hexdigest()
        )

    def update(self, A):
        """Factorize the matrix A, the factorization will automatically be used if the same matrix A is passed to the
        solve method. This will drastically increase the speed of solve, if solve is called more than once for the
        same matrix A

        Args:
            A: sparse square CSR or CSC matrix (:py:class:`scipy.sparse.csr.csr_matrix`)
        """
        self.A = self._check_A(A)

        if self.A.nnz > self.size_limit_storage:
            self._factorized_A = self._hash_csr_matrix(self.A)
        else:
            self._factorized_A = self.A.copy()

        self._phase = 12  # Do 1) analysis, symbolic factorization, and 2) numerical factorization
        self._iparm[11] = 0
        self._call_pardiso()
        self._phase = 33  # Set solver ready for 3) solving with iterative refinement

    def solve(self, b, x0=None, trans="N"):
        """solve Ax=b for x

        Args:
            A (scipy.sparse.csr.csr_matrix): sparse square CSR matrix , CSC matrix also possible
            b (numpy.ndarray): right-hand side(s), b.shape[0] needs to be the same as A.shape[0]
            x0 (unused)
            trans (optional): Indicate which system to solve (Normal, Transposed, or Hermitian transposed)

        Returns:
            Solution of the system of linear equations, same shape as input b
        """
        if b.shape[0] != self.A.shape[0]:
            raise ValueError("Shape mismatch: Matrix A {self.A.shape} and array b {b.shape}")
        b = self._check_b(b)

        if trans not in ["N", "T", "H"]:
            raise TypeError("Only N, T, or H transposition is possible")

        """ Conversion table, T = transpose mode, C = conjugate mode  (T + C = H)
         mode | CSR | CSC
        ------+-----+-----
           N  | - - | T -
           T  | T - | - -
           H  | T C | - C
        """
        conjugate_mode = trans == "H"
        transpose_mode = not (trans == "N")
        if self._solve_transposed:
            transpose_mode = not transpose_mode

        # Set pardiso transposition mode, 0: normal, 1: conjugate transpose, 2: transposed
        if transpose_mode:
            if conjugate_mode:
                self._iparm[11] = 1
            else:
                self._iparm[11] = 2
        else:
            if conjugate_mode:
                b = np.conj(b)  # Transposed-conjugate-transposed matrix == conjugated matrix
            self._iparm[11] = 0

        x = self._call_pardiso(b=b)

        if conjugate_mode and not transpose_mode:
            return np.conj(x)  # Conjugate back
        else:
            return x

    def clear_factorization(self):
        self._factorized_A = sps.csr_matrix((0, 0))

    def __del__(self):
        try:
            self.clear_factorization()
            A = sps.csr_matrix((0, 0))
            b = np.zeros(0)
            self._phase = -1
            self._call_pardiso(A, b)
            self._phase = 13
        except ImportError:
            # To prevent ImportError: sys.meta_path is None, Python is likely shutting down
            pass


# ------------------------------------ LU Solver -----------------------------------
try:
    from scikits.umfpack import splu  # UMFPACK solver; this one is faster and has identical interface
except ImportError:
    from scipy.sparse.linalg import splu


class SolverSparseLU(LinearSolver):
    """Solver for sparse (square) matrices using an LU decomposition.

    Internally, `scipy` uses the SuperLU library, which is relatively slow. It may be sped up using the Python package
    ``scikit-umfpack``, which scipy` is able to use, but must be installed by the user.

    References:
      - `Scipy LU <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html>`_
      - `Scipy UMFPACK <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.use_solver.html>`_
    """

    def update(self, A):
        r"""Factorize the matrix as :math:`\mathbf{A}=\mathbf{L}\mathbf{U}`, where :math:`\mathbf{L}` is a lower
        triangular matrix and :math:`\mathbf{U}` is upper triangular.
        """
        self.iscomplex = matrix_is_complex(A)
        self.inv = splu(A)
        return self

    def solve(self, rhs, x0=None, trans="N"):
        r"""Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{U}^{-1}\mathbf{L}^{-1}\mathbf{b}`.

        Adjoint system solves the linear system of equations :math:`\mathbf{A}^\text{H}\mathbf{x} = \mathbf{b}` by
        forward and backward substitution of :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{U}^{-\text{H}}\mathbf{b}`
        """
        if trans not in ["N", "T", "H"]:
            raise TypeError("Only N, T, or H transposition is possible")
        return self.inv.solve(rhs, trans=trans)


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
    """Solver for positive-definite Hermitian matrices using a Cholesky factorization.

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
        r"""Factorize the matrix using Cholmod. In case the matrix :math:`\mathbf{A}` is non-Hermitian, the
        system of equations is solved in a least-squares sense:
        :math:`\min \left| \mathbf{A}\mathbf{x} - \mathbf{b} \right|^2`.
        The solution of this minimization is
        :math:`\mathbf{x}=(\mathbf{A}^\text{H}\mathbf{A})^{-1}\mathbf{A}^\text{H}\mathbf{b}`.
        """
        if not sps.issparse(A):
            warnings.warn(f"{type(self).__name__}: Efficiency warning: Matrix should be sparse", 
                          SparseEfficiencyWarning)
            self.A = sps.csc_matrix(A)
        else:
            self.A = A

        if not hasattr(self, "inv"):
            self.inv = cholmod.analyze(self.A)

        # Do the Cholesky factorization
        self.inv.cholesky_inplace(self.A)

        return self

    def solve(self, rhs, x0=None, trans="N"):
        r"""Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{L}^{-1}\mathbf{b}` in case of an
        Hermitian matrix.

        The right-hand-side :math:`\mathbf{b}` can be of size ``(N)`` or ``(N, K)``, where ``N`` is the size of matrix
        :math:`\mathbf{A}` and ``K`` is the number of right-hand sides.

        """
        if trans not in ["N", "T", "H"]:
            raise TypeError("Only N, T, or H transposition is possible")
        if trans == "T":
            return self.inv(rhs.conj()).conj()
        else:
            return self.inv(rhs)


# ------------------------------------ Cholesky Solver cvxopt -----------------------------------
try:  # CVXOPT cholmod
    import cvxopt
    import cvxopt.cholmod

    _has_cvxopt_cholmod = True
except ImportError:
    _has_cvxopt_cholmod = False


class SolverSparseCholeskyCVXOPT(LinearSolver):
    """Solver for positive-definite Hermitian matrices using a Cholesky factorization.

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
        r"""Factorize the matrix using CVXOPT's Cholmod as :math:`\mathbf{A}=\mathbf{L}\mathbf{L}^\text{H}`."""
        if not isinstance(A, cvxopt.spmatrix):
            if not isinstance(A, sps.coo_matrix):
                Kcoo = sps.coo_matrix(A)
                warnings.warn(f"{type(self).__name__}: Efficiency warning: CVXOPT spmatrix must be used", 
                              SparseEfficiencyWarning)
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

    def solve(self, rhs, x0=None, trans="N"):
        r"""Solves the linear system of equations :math:`\mathbf{A} \mathbf{x} = \mathbf{b}` by forward and backward
        substitution of :math:`\mathbf{x} = \mathbf{L}^{-\text{H}}\mathbf{L}^{-1}\mathbf{b}`."""
        if trans not in ["N", "T", "H"]:
            raise TypeError("Only N, T, or H transposition is possible")
        if rhs.dtype != self._dtype:
            warnings.warn(
                f"{type(self).__name__}: Type warning: rhs value type ({rhs.dtype}) is converted to {self._dtype}"
            )
        if trans == "T":
            B = cvxopt.matrix(rhs.conj().astype(self._dtype))
        else:
            B = cvxopt.matrix(rhs.astype(self._dtype))
        nrhs = 1 if rhs.ndim == 1 else rhs.shape[1]

        cvxopt.cholmod.solve(self.inv, B, nrhs=nrhs)

        x = np.array(B).flatten() if rhs.ndim == 1 else np.array(B)
        return x.conj() if trans == "T" else x
