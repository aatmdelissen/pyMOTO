""" Specialized linear algebra modules """
import warnings

from pymodular.core_objects import Module
from pymodular.dyadcarrier import DyadCarrier
import numpy as np
import scipy.sparse as sps
import scipy.linalg as spla # Dense matrix solvers
try:
    from scikits.umfpack import splu # UMFPACK solver; this one is faster
except ImportError:
    from scipy.sparse.linalg import splu

class LinearSolver:
    defined=True
    _err_msg = ""
    def __init__(self, A=None):
        if A is not None:
            self.update(A)

    def update(self, A):
        """ Updates with a new matrix of the same structure
        :param A: The new matrix
        :return: self
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def solve(self, rhs):
        """ Solves A x = rhs
        :param rhs: Right hand side
        :return: Solution vector x
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    def adjoint(self, rhs):
        """Solves A^H x = rhs in case of complex matrix or A^T x = rhs for a real-valued matrix
        :param rhs: Right hand side
        :return: Solution vector x
        """
        raise NotImplementedError(f"Solver not implemented {self._err_msg}")

    @staticmethod
    def residual(A, x, b):
        """ Calculates the residual || A x - b || / || b ||
        :param A: Matrix
        :param x: Solution
        :param b: Right-hand side
        :return: Residual value
        """
        return np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b)


class SolverDiagonal(LinearSolver):
    def update(self, A):
        self.diag = A.diagonal()
        return self
    def solve(self, rhs):
        return rhs/self.diag
    def adjoint(self, rhs):
        return rhs/(self.diag.conj())


# Dense QR solver
class SolverDenseQR(LinearSolver):
    def update(self, A):
        self.A = A
        self.q,self.r = spla.qr(A)
        return self
    def solve(self, rhs):
        return spla.solve_triangular(self.r,self.q.T.conj()@rhs)
    def adjoint(self, rhs):
        return self.q@spla.solve_triangular(self.r, rhs, trans='C')


# Dense LU solver
class SolverDenseLU(LinearSolver):
    def update(self, A):
        self.p, self.l, self.u = spla.lu(A)
        return self
    def solve(self, rhs):
        return spla.solve_triangular(self.u, spla.solve_triangular(self.l, self.p.T@rhs, lower=True))
    def adjoint(self, rhs):
        return self.p@spla.solve_triangular(self.l, spla.solve_triangular(self.u, rhs, trans='C'), lower=True, trans='C')# TODO permutation


# Dense Cholesky solver
class SolverDenseCholesky(LinearSolver):
    """ Only for Hermitian positive-definite matrix """
    def __init__(self, *args, **kwargs):
        self.backup_solver = SolverDenseLDL()
        self.success=None
        super().__init__(*args, **kwargs)
    def update(self, A):
        try:
            self.u = spla.cholesky(A)
            self.success = True
        except np.linalg.LinAlgError as err:
            warnings.warn(f"{type(self).__name__}: {err} -- using {type(self.backup_solver).__name__} instead")
            self.backup_solver.update(A)
            self.success = False
        return self
    def solve(self, rhs):
        if self.success:
            return spla.solve_triangular(self.u, spla.solve_triangular(self.u, rhs, trans='C'))
        else:
            return self.backup_solver.solve(rhs)
    def adjoint(self, rhs):
        if self.success:
            return self.solve(rhs)
        else:
            return self.backup_solver.adjoint(rhs)


# Dense LDL solver
class SolverDenseLDL(LinearSolver):
    def __init__(self, *args, hermitian=None, **kwargs):
        self.hermitian = hermitian
        super().__init__(*args, **kwargs)
    def update(self, A):
        if self.hermitian is None:
            self.hermitian = matrix_is_hermitian(A)
        self.l, self.d, self.p = spla.ldl(A, hermitian=self.hermitian)
        self.diagonald = np.allclose(self.d, np.diag(np.diag(self.d)))
        if self.diagonald: # Exact diagonal
            d1 = np.diag(1/np.diag(self.d))
        else:
            d1 = np.linalg.inv(self.d) # TODO, this could be improved
        self.dinv = lambda b: d1@b
        self.dinvH = lambda b: (d1.conj().T)@b
        self.lp = self.l[self.p, :]
        return self
    def solve(self, rhs):
        u1 = spla.solve_triangular(self.lp, rhs[self.p], lower=True, unit_diagonal=True)
        u2 = self.dinv(u1)
        u = np.zeros_like(rhs)
        u[self.p] = spla.solve_triangular(self.lp, u2, trans='C' if self.hermitian else 'T', lower=True, unit_diagonal=True)
        return u
    def adjoint(self, rhs):
        if not self.hermitian:
            u1 = spla.solve_triangular(self.lp, rhs[self.p].conj(), lower=True, unit_diagonal=True).conj()
        else:
            u1 = spla.solve_triangular(self.lp, rhs[self.p], lower=True, unit_diagonal=True)
        u2 = self.dinvH(u1)
        u = np.zeros_like(rhs)
        u[self.p] = spla.solve_triangular(self.lp, u2, trans='C', lower=True, unit_diagonal=True)
        return u

# ----- SPARSE SOLVERS -----
import os
import sys
import glob
import ctypes
import hashlib
from ctypes.util import find_library

import numpy as np
import scipy.sparse as sp
from scipy.sparse import SparseEfficiencyWarning

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
        roots = [sys.prefix, os.environ['MKLROOT']]
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

__libmkl = _find_libmkl()

if __libmkl is not None:
    class SolverSparsePardiso(LinearSolver):
        """
        Python interface to the Intel MKL PARDISO library for solving large sparse linear systems of equations Ax=b.
        Pardiso documentation: https://software.intel.com/en-us/node/470282
        --- Basic usage ---
        matrix type: real (float64) and nonsymetric
        methods: solve, factorize
        - use the "solve(A,b)" method to solve Ax=b for x, where A is a sparse CSR (or CSC) matrix and b is a numpy array
        - use the "factorize(A)" method first, if you intend to solve the system more than once for different right-hand
          sides, the factorization will be reused automatically afterwards
        --- Advanced usage ---
        methods: get_iparm, get_iparms, set_iparm, set_matrix_type, set_phase
        - additional options can be accessed by setting the iparms (see Pardiso documentation for description)
        - other matrix types can be chosen with the "set_matrix_type" method. complex matrix types are currently not
          supported. pypardiso is only teste for mtype=11 (real and nonsymetric)
        - the solving phases can be set with the "set_phase" method
        - The out-of-core (OOC) solver either fails or crashes my computer, be careful with iparm[60]
        --- Statistical info ---
        methods: set_statistical_info_on, set_statistical_info_off
        - the Pardiso solver writes statistical info to the C stdout if desired
        - if you use pypardiso from within a jupyter notebook you can turn the statistical info on and capture the output
          real-time by wrapping your call to "solve" with wurlitzer.sys_pipes() (https://github.com/minrk/wurlitzer,
          https://pypi.python.org/pypi/wurlitzer/)
        - wurlitzer dosen't work on windows, info appears in notebook server console window if used from jupyter notebook
        --- Memory usage ---
        methods: remove_stored_factorization, free_memory
        - remove_stored_factorization can be used to delete the wrapper's copy of matrix A
        - free_memory releases the internal memory of the solver
        """
    
        def __init__(self, mtype=2, phase=13, size_limit_storage=5e7):
            """
            mtype: 1  real and structurally symmetric
                   2  real and symmetric positive definite
                  -2  real and symmetric indefinite
                   3  complex and structurally symmetric 
                   4  complex and Hermitian positive definite
                  -4  complex and Hermitian indefinite
                   6  complex and symmetric
                   11 real and nonsymmetric
                   13 complex and nonsymmetric
            :param mtype: 
            :param phase: 
            :param size_limit_storage: 
            """
    
            self.libmkl = __libmkl
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
    
            self.mtype = mtype
            self.phase = phase
            self.msglvl = False
    
            self.factorized_A = sp.csr_matrix((0, 0))
            self.size_limit_storage = size_limit_storage
            self._solve_transposed = False
            self._is_factorized = False
            
            self.set_iparm(0, 1) # Use non-default values
            self.set_iparm(34, 1) # Zero-based indexing, as is done in Python
    
        def update(self, A):
            """
            Factorize the matrix A, the factorization will automatically be used if the same matrix A is passed to the
            solve method. This will drastically increase the speed of solve, if solve is called more than once for the
            same matrix A
            --- Parameters ---
            A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix)
            """
            if self.mtype in {-2, 2, 6}:
                A = sp.triu(A, format='csr')
            if not sp.isspmatrix_csr(A):
                warnings.warn(f'PyPardiso requires CSR matrix format, not {type(A).__name__}', SparseEfficiencyWarning)
                A = A.tocsr()
            self.A = A
            self._check_A(A)
    
            if A.nnz > self.size_limit_storage:
                self.factorized_A = self._hash_csr_matrix(A)
            else:
                self.factorized_A = A.copy()
    
            self.set_phase(12)
            b = np.zeros((A.shape[0], 1))
            self._call_pardiso(A, b)
            self._is_factorized = True
    
        def solve(self, b):
            """
            solve Ax=b for x
            --- Parameters ---
            A: sparse square CSR matrix (scipy.sparse.csr.csr_matrix), CSC matrix also possible
            b: numpy ndarray
               right-hand side(s), b.shape[0] needs to be the same as A.shape[0]
            --- Returns ---
            x: numpy ndarray
               solution of the system of linear equations, same shape as input b
            """
    
            # self._check_A(A)
            b = self._check_b(self.A, b)
    
            if self._is_factorized:
                self.set_phase(33)
            else:
                self.set_phase(13)
    
            x = self._call_pardiso(self.A, b)
    
            return x
    
        def adjoint(self, b):
            b = self._check_b(self.A, b)
            self.set_iparm(11, 1) # Adjoint solver
            x = self._call_pardiso(self.A, b)
            self.set_iparm(11, 0) # Revert back to normal solver
            return x
    
        def _is_already_factorized(self, A):
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
    
            if sp.isspmatrix_csr(A):
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
    
            if A.dtype != np.float64:
                raise TypeError('PyPardiso currently only supports float64, but matrix A has dtype: {}'.format(A.dtype))
    
        def _check_b(self, A, b):
            if sp.isspmatrix(b):
                warnings.warn('PyPardiso requires the right-hand side b to be a dense array for maximum efficiency',
                              SparseEfficiencyWarning)
                b = b.todense()
    
            # pardiso expects fortran (column-major) order if b is a matrix
            if b.ndim == 2:
                b = np.asfortranarray(b)
    
            if b.shape[0] != A.shape[0]:
                raise ValueError("Dimension mismatch: Matrix A {} and array b {}".format(A.shape, b.shape))
    
            if b.dtype != np.float64:
                if b.dtype in [np.float16, np.float32, np.int16, np.int32, np.int64]:
                    warnings.warn("Array b's data type was converted from {} to float64".format(str(b.dtype)),
                                  PyPardisoWarning)
                    b = b.astype(np.float64)
                else:
                    raise TypeError('Dtype {} for array b is not supported'.format(str(b.dtype)))
    
            return b
    
        def _call_pardiso(self, A, b):
    
            x = np.zeros_like(b)
            pardiso_error = ctypes.c_int32(0)
            c_int32_p = ctypes.POINTER(ctypes.c_int32)
            c_float64_p = ctypes.POINTER(ctypes.c_double)
    
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
                              A.data.ctypes.data_as(c_float64_p),  # A -> non-zero entries in matrix
                              ia.ctypes.data_as(c_int32_p),  # ia -> csr-indptr
                              ja.ctypes.data_as(c_int32_p),  # ja -> csr-indices
                              self.perm.ctypes.data_as(c_int32_p),  # perm -> empty
                              ctypes.byref(ctypes.c_int32(1 if b.ndim == 1 else b.shape[1])),  # nrhs
                              self.iparm.ctypes.data_as(c_int32_p),  # iparm-array
                              ctypes.byref(ctypes.c_int32(self.msglvl)),  # msg-level -> 1: statistical info is printed
                              b.ctypes.data_as(c_float64_p),  # b -> right-hand side vector/matrix
                              x.ctypes.data_as(c_float64_p),  # x -> output
                              ctypes.byref(pardiso_error))  # pardiso error
    
            if pardiso_error.value != 0:
                raise PyPardisoError(pardiso_error.value)
            else:
                return np.ascontiguousarray(x)  # change memory-layout back from fortran to c order
    
        def get_iparms(self):
            """Returns a dictionary of iparms"""
            return dict(enumerate(self.iparm, 1))
    
        def get_iparm(self, i):
            """Returns the i-th iparm (1-based indexing)"""
            return self.iparm[i]
    
        def set_iparm(self, i, value):
            """set the i-th iparm to 'value' (1-based indexing)
            https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html
            """
            if i not in {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 2426, 27, 29, 30, 33, 34, 35, 36, 38, 42, 55, 59, 62}:
                warnings.warn('{} is no input iparm. See the Pardiso documentation.'.format(value), PyPardisoWarning)
            self.iparm[i] = value
    
        def print_iparm(self):
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
                if v!=0:
                    print(f"{i}: {v} ({k})")
    
        def set_matrix_type(self, mtype):
            """Set the matrix type (see Pardiso documentation)"""
            self.mtype = mtype
    
        def set_statistical_info_on(self):
            """Display statistical info (appears in notebook server console window if pypardiso is
            used from jupyter notebook, use wurlitzer to redirect info to the notebook)"""
            self.msglvl = 1
    
        def set_statistical_info_off(self):
            """Turns statistical info off"""
            self.msglvl = 0
    
        def set_phase(self, phase):
            """Set the phase(s) for the solver. See the Pardiso documentation for details."""
            self.phase = phase
    
        def remove_stored_factorization(self):
            """removes the stored factorization, this will free the memory in python, but the factorization in pardiso
            is still accessible with a direct call to self._call_pardiso(A,b) with phase=33"""
            self.factorized_A = sp.csr_matrix((0, 0))
    
        def free_memory(self, everything=False):
            """release mkl's internal memory, either only for the factorization (ie the LU-decomposition) or all of
            mkl's internal memory if everything=True"""
            self.remove_stored_factorization()
            A = sp.csr_matrix((0, 0))
            b = np.zeros(0)
            self.set_phase(-1 if everything else 0)
            self._call_pardiso(A, b)
            self.set_phase(13)
else:
    class SolverSparsePardiso(LinearSolver):
        defined = False
        _err_msg = "-- Intel OneAPI not installed"


class PyPardisoWarning(UserWarning):
    pass


class PyPardisoError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return ('The Pardiso solver failed with error code {}. '
                'See Pardiso documentation for details.'.format(self.value))


class SolverSparseLU(LinearSolver):
    def update(self, A):
        self.iscomplex = np.iscomplexobj(A)
        self.inv = splu(A)
        return self
    def solve(self, rhs):
        return self.inv.solve(rhs)
    def adjoint(self, rhs):
        return self.inv.solve(rhs, trans=('H' if self.iscomplex else 'T'))


# Sparse cholesky solver
try:  # SCIKIT CHOLMOD 
    from sksparse import cholmod # Sparse cholesky solver (CHOLMOD)
    class SolverSparseCholeskyScikit(LinearSolver):
        def update(self, A):
            if not hasattr(self, 'inv'):
                self.inv = cholmod.analyze(A)
            self.iscomplex = np.iscomplexobj(A)
            self.inv.cholesky_inplace(A)
            return self
        def solve(self, rhs):
            return self.inv(rhs)
        def adjoint(self, rhs):
            return self.solve(rhs.conj()).conj()
except ImportError: 
    class SolverSparseCholeskyScikit(LinearSolver):
        defined=False
        _err_msg = "-- scikit-sparse not installed"
        
        
try: # CVXOPT cholmod
    import cvxopt
    import cvxopt.cholmod
    class SolverSparseCholeskyCVXOPT(LinearSolver):
        def update(self, A):
            if not isinstance(A, sps.coo_matrix):
                Kcoo = A.tocoo()
            else:
                Kcoo = A
            K = cvxopt.spmatrix(Kcoo.data, Kcoo.row.astype(int), Kcoo.col.astype(int))
            if not hasattr(self, 'inv'):
                self.inv = cvxopt.cholmod.symbolic(K)
            cvxopt.cholmod.numeric(K, self.inv)
            return self
        def solve(self, rhs):
            B = cvxopt.matrix(rhs)
            nrhs = 1 if rhs.ndim == 1 else rhs.shape[1]
            cvxopt.cholmod.solve(self.inv, B, nrhs=nrhs)
            return np.array(B).flatten() if nrhs == 1 else np.array(B)
        def adjoint(self, rhs):
            return self.solve(rhs.conj()).conj()
except ImportError:
    class SolverSparseCholeskyCVXOPT(LinearSolver):
        defined = False
        _err_msg = "-- cvxopt not installed"


def matrix_is_diagonal(A):
    # TODO: This could be improved to check other sparse matrix types as well
    if sps.issparse(A):
        return isinstance(A, sps.dia_matrix) and len(A.offsets)==1 and A.offsets[0]==0
    else:
        return np.allclose(A, np.diag(np.diag(A)))


def matrix_is_symmetric(A):
    if sps.issparse(A):
        return np.allclose((A-A.T).data, 0)
    else:
        return np.allclose(A, A.T)


def matrix_is_hermitian(A):
    if not np.iscomplexobj(A):
        return matrix_is_symmetric(A)
    else:
        if sps.issparse(A):
            return np.allclose((A-A.T.conj()).data, 0)
        else:
            return np.allclose(A, A.T.conj())


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
            sps.SparseEfficiencyWarning("Only a dense version of QR solver is available") # TODO
        return SolverDenseQR()  

    if isdiagonal is None: # Check if matrix is diagonal
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
        if ishermitian is None:
            # Detect if the matrix is hermitian
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
        if ishermitian:
            # Check if diagonal is all positive or all negative
            if np.alltrue(A.diagonal()>0) or np.alltrue(A.diagonal()<0):
                if SolverSparseCholeskyScikit.defined:
                    return SolverSparseCholeskyScikit()
                elif SolverSparseCholeskyPardiso.defined:
                    return SolverSparseCholeskyPardiso()
                elif SolverSparseCholeskyCVXOPT.defined:
                    return SolverSparseCholeskyCVXOPT()
        
        return SolverSparseLU()
            
    else: # Dense
        if ishermitian:
            # Check if diagonal is all positive or all negative
            if np.alltrue(A.diagonal()>0) or np.alltrue(A.diagonal()<0):
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
                warnings.warn("This one has not been checked yet!") # TODO
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


