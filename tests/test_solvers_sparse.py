import inspect
import pathlib  # For importing files
import sys
import unittest
from math import isclose

import numpy as np
import numpy.testing as npt
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
from scipy.io import mmread  # For importing files

import pymoto as pym

try:
    import cvxopt
    _has_cvxopt = True
except ImportError:
    sys.stdout.write("Skipping all tests with CVXOPT matrix...\n")
    _has_cvxopt = False

np.random.seed(0)  # Set seed for repeatability


def calculate_eigenvalues(A):
    """ Calculates the eigenvalues of a matrix to check positive-definiteness """
    return sp.linalg.eigvals(A.toarray())


def load_matrix(filen):
    """ Loads a matrix from file """
    curdir = pathlib.Path(__file__).parent.resolve()
    return mmread(pathlib.Path(curdir, "data", filen))


def generate_real_diagonal(N):
    """ Generates a real diagonal matrix of size N """
    return spsp.spdiags(np.random.rand(N), 0, N, N).tocoo()  # All positive


def generate_complex_diagonal(N):
    """ Generates a complex diagonal matrix of size N """
    return spsp.spdiags(np.random.rand(N) + 1j*np.random.rand(N), 0, N, N).tocoo()


def generate_symm_indef():
    """ Generates a real symmetric indefinite matrix """
    A = load_matrix("zenios.mtx.gz")
    sigma = -0.1  # To prevent exact singularity
    As = A - sigma*spsp.eye(A.shape[0])
    # Calculate eigenvalues
    # solver = pym.solvers.SolverSparseLU().update(As)
    # opinv = spspla.LinearOperator(A.shape, matvec=solver.solve, rmatvec=solver.adjoint)
    # w, _ = spspla.eigsh(A, sigma=sigma, k=100, OPinv=opinv, ncv=200)]
    return As.tocoo()


def generate_dynamic(complex=False, indef=False):
    """ Generate a dynamic stiffness matrix, either complex or not, and indefinite or not"""
    K = load_matrix("bcsstk26.mtx.gz")
    M = load_matrix("bcsstm26.mtx.gz")
    solver = pym.solvers.SolverSparseLU().update(K.tocsc())
    opinv = spspla.LinearOperator(K.shape, matvec=solver.solve, rmatvec=lambda b: solver.adjoint(b, trans='H'))
    w, _ = spspla.eigsh(K, M=M, sigma=0, k=100, OPinv=opinv, ncv=200)
    eigfreqs = np.sqrt(w)
    if indef:
        freq = (eigfreqs[0] + eigfreqs[1])/2
    else:
        freq = eigfreqs[0]/2
    if complex:
        Ks = K + freq*1j*(0.01*K + 0.01*M) - freq**2 * M
    else:
        Ks = K - freq**2 * M
    return Ks.tocoo()


def generate_saddlepoint():
    K = load_matrix("bcsstk26.mtx.gz")
    M = load_matrix("bcsstm26.mtx.gz")
    solver = pym.solvers.SolverSparseLU().update(K.tocsc())
    opinv = spspla.LinearOperator(K.shape, matvec=solver.solve, rmatvec=lambda b: solver.adjoint(b, trans='H'))
    w, V = spspla.eigsh(K, M=M, sigma=0, k=100, OPinv=opinv, ncv=200)
    wi, vi = w[0], V[:, [0]]
    Mv = M@vi

    return spsp.bmat([[K-wi*M, -Mv],
                      [-Mv.T, None]]).tocoo()


""" Perpare the set of matrices we want to test """
# Real-valued matrices
mat_real_spdiag = spsp.spdiags(np.random.rand(100), 0, 100, 100)  # Diagonal (spdiag format)
mat_real_diagonal = generate_real_diagonal(100)  # Diagonal (coo format)
mat_real_symm_pos_def = load_matrix("bcsstk14.mtx.gz")  # Symmetric positive definite
mat_real_symm_pos_def_dynamic = generate_dynamic(complex=False, indef=False)  # Symmetric positive definite
mat_real_symm_indef = generate_symm_indef()  # Symmetric indefinite
mat_real_symm_indef_dynamic = generate_dynamic(complex=False, indef=True)  # Symmetric indefinite
mat_real_symm_saddle = generate_saddlepoint()  # Indefinite saddlepoinnt
mat_real_asymm = load_matrix("impcol_a.mtx.gz")  # Asymmetric

# Complex-valued matrices
mat_complex_spdiag = mat_real_spdiag + 1j*mat_real_spdiag  # Diagonal (spdiag format)
mat_complex_diagonal = generate_complex_diagonal(100)  # Diagonal (coo format)
mat_complex_symm_pos_def_dynamic = generate_dynamic(complex=True, indef=False)  # Symmetric positive definite
mat_complex_symm_indef_dynamic = generate_dynamic(complex=True, indef=True)  # Symmetric indefinite
mat_complex_hermitian_1 = load_matrix("mhd1280b.mtx.gz")  # Hermitian
mat_complex_hermitian_indef = mat_complex_hermitian_1 - spsp.eye(*mat_complex_hermitian_1.shape)
mat_complex_hermitian_pos_def = mat_complex_hermitian_1 + spsp.eye(*mat_complex_hermitian_1.shape)

""" ------------------ TEST UTILITY FUNCTIONS -------------------- """


class TestGenericUtility(unittest.TestCase):
    """ Generic test class for testing functions on matrix properties that return True/False """
    data = []  # The data to test: list of tuples with (matrix, True/False property)
    fn = None  # The test function to use, which return the True/False property

    @classmethod
    def setUpClass(cls) -> None:
        """ Skip test if no function is given """
        if cls.fn is None:
            raise unittest.SkipTest(f"Skipping test {cls}")

    def get_tags(self):
        """ Generate tags for all the matrices to use in messaging"""
        src = inspect.getsource(type(self))
        start = src.find('data')
        start += src[start:].find('=') + 1
        finish = start + src[start:].find(']')

        def parse_str(t: str):
            rmv = ['\n', ' ', '(', ',', 'True', 'False']
            for c in rmv:
                t = t.replace(c, '')
            return t

        return [parse_str(t) for t in src[start:finish].strip(' []').split(')')]

    def test_all_matrices(self):
        """ Test all the matrices """
        tags = self.get_tags()
        for i, dat in enumerate(self.data):
            A = dat[0]
            res = dat[1]
            # Numpy dense matrix
            with self.subTest(msg=f"Dense - {tags[i]}"):
                # sys.stdout.write(f"Test {self.fn.__name__} for dense matrix {tags[i]}\n")
                self.assertEqual(self.fn(A.toarray()), res)

            # Scipy COO / spdiag matrix
            with self.subTest(msg=f"{type(A)} - {tags[i]}"):
                # sys.stdout.write(f"Test {self.fn.__name__} for {type(A)}-matrix {tags[i]}\n")
                self.assertEqual(self.fn(A), res)

            # Scipy CSR matrix
            with self.subTest(msg=f"CSR - {tags[i]}"):
                # sys.stdout.write(f"Test {self.fn.__name__} for CSR-matrix {tags[i]}\n")
                self.assertEqual(self.fn(A.tocsr()), res)

            # Scipy CSC matrix
            with self.subTest(msg=f"CSC - {tags[i]}"):
                # sys.stdout.write(f"Test {self.fn.__name__} for CSC-matrix {tags[i]}\n")
                self.assertEqual(self.fn(A.tocsc()), res)

            # CVXOPT spmatrix
            if _has_cvxopt:
                with self.subTest(msg=f"CVXOPT - {tags[i]}"):
                    # sys.stdout.write(f"Test {self.fn.__name__} for CVXOPT-matrix {tags[i]}\n")
                    if isinstance(A, spsp.coo_matrix):
                        Acvx = cvxopt.spmatrix(A.data, A.row.astype(int), A.col.astype(int))
                    self.assertEqual(self.fn(Acvx), res)


class TestIsComplex(TestGenericUtility):
    fn = staticmethod(pym.solvers.matrix_is_complex)
    data = [
        (mat_real_diagonal, False),
        (mat_real_spdiag, False),
        (mat_real_symm_pos_def, False),
        (mat_real_symm_pos_def_dynamic, False),
        (mat_real_symm_indef, False),
        (mat_real_symm_indef_dynamic, False),
        (mat_real_asymm, False),
        (mat_complex_diagonal, True),
        (mat_complex_spdiag, True),
        (mat_complex_symm_pos_def_dynamic, True),
        (mat_complex_symm_indef_dynamic, True),
        (mat_complex_hermitian_1, True),
        (mat_complex_hermitian_indef, True),
        (mat_complex_hermitian_pos_def, True),
    ]


class TestIsDiagonal(TestGenericUtility):
    fn = staticmethod(pym.solvers.matrix_is_diagonal)
    data = [
        (mat_real_diagonal, True),
        (mat_real_spdiag, True),
        (mat_real_symm_pos_def, False),
        (mat_real_symm_pos_def_dynamic, False),
        (mat_real_symm_indef, False),
        (mat_real_symm_indef_dynamic, False),
        (mat_real_asymm, False),
        (mat_complex_diagonal, True),
        (mat_complex_spdiag, True),
        (mat_complex_symm_pos_def_dynamic, False),
        (mat_complex_symm_indef_dynamic, False),
        (mat_complex_hermitian_1, False),
        (mat_complex_hermitian_indef, False),
        (mat_complex_hermitian_pos_def, False),
    ]


class TestIsSymmetric(TestGenericUtility):
    fn = staticmethod(pym.solvers.matrix_is_symmetric)
    data = [
        (mat_real_diagonal, True),
        (mat_real_spdiag, True),
        (mat_real_symm_pos_def, True),
        (mat_real_symm_pos_def_dynamic, True),
        (mat_real_symm_indef, True),
        (mat_real_symm_indef_dynamic, True),
        (mat_real_asymm, False),
        (mat_complex_diagonal, True),
        (mat_complex_spdiag, True),
        (mat_complex_symm_pos_def_dynamic, True),
        (mat_complex_symm_indef_dynamic, True),
        (mat_complex_hermitian_1, False),
        (mat_complex_hermitian_indef, False),
        (mat_complex_hermitian_pos_def, False),
    ]


class TestIsHermitian(TestGenericUtility):
    fn = staticmethod(pym.solvers.matrix_is_hermitian)
    data = [
        (mat_real_diagonal, True),
        (mat_real_spdiag, True),
        (mat_real_symm_pos_def, True),
        (mat_real_symm_pos_def_dynamic, True),
        (mat_real_symm_indef, True),
        (mat_real_symm_indef_dynamic, True),
        (mat_real_asymm, False),
        (mat_complex_diagonal, False),
        (mat_complex_spdiag, False),
        (mat_complex_symm_pos_def_dynamic, False),
        (mat_complex_symm_indef_dynamic, False),
        (mat_complex_hermitian_1, True),
        (mat_complex_hermitian_indef, True),
        (mat_complex_hermitian_pos_def, True),
    ]


""" ------------------ TEST THE SPARSE SOLVERS -------------------- """
# SOLVERS:
# - scipy/umfpack lu
# - scikit cholmod
# - cvxopt cholmod
# - intel mkl pardiso


class GenericTestSolvers(unittest.TestCase):
    """ Generic test runner for any solver and list of matrices """
    solver = None  # The solver to use
    matrices = []  # The list of matrices the solver should be able to handle
    kwargs = dict()

    @classmethod
    def setUpClass(cls) -> None:
        """ Skip test if the solver is not available """
        if cls.solver is None:
            raise unittest.SkipTest(f"No solver given for {cls.__name__}")
        if not cls.solver.defined:
            raise unittest.SkipTest(f"Solver \"{cls.solver.__name__}\" not available, probably lacking third-party libraries")

    def get_tags(self):
        """ Get the matrix tags for messages """
        src = inspect.getsource(type(self))
        start = src.find('matrices')
        start += src[start:].find('=') + 1
        finish = start + src[start:].find(']')
        tags_orig = src[start:finish].strip(' []').split(',')
        tags_proc = [t.strip(' ').replace('\n', '').replace(' ', '') for t in tags_orig]
        # Remove commented or empty
        tags_final = []
        for t in tags_proc:
            if len(t) == 0:
                continue
            if t[0] == '#':
                continue
            tags_final.append(t)
        return tags_final

    def run_solver(self, solver, A, b):
        """ Run the actual test for given solver, matrix and right-hand-side """
        # Reference solution
        mats = dict()
        mats['N'] = A
        mats['T'] = A.T
        mats['H'] = A.conj().T
        solver.update(A)

        for k, Ai in mats.items():
            print(f'Run mode = {k}')
            xref = pym.solvers.SolverSparseLU(Ai).solve(b)

            # Calculate the solution and the adjoint solution
            x = solver.solve(b, trans=k)

            # Check residual and solution
            r = np.linalg.norm(Ai @ x - b) / np.linalg.norm(b)
            self.assertTrue(r < 1e-4, msg=f"Residual is {r}")
            npt.assert_allclose(np.real(x), np.real(xref), atol=1e-10, rtol=1e-5)
            if np.iscomplexobj(x) or np.iscomplexobj(xref):
                npt.assert_allclose(np.imag(x), np.imag(xref), atol=1e-8, rtol=1e-5)

    def run_sparse_tests(self, solver, Acoo, rhs):
        """ Run the sparse tests for different matrix types """
        self.run_solver(solver, Acoo, rhs)
        self.run_solver(solver, Acoo.tocsr(), rhs)
        self.run_solver(solver, Acoo.tocsc(), rhs)

    def test_all_matrices(self):
        """ Run the tests on all given matrices """
        tags = self.get_tags()
        for A, t in zip(self.matrices, tags):
            """ Test for different type of right-hand-sides """
            sys.stdout.write(f"Test \"{self.solver.__name__}\" for matrix \"{t}\"\n")
            N = A.shape[0]
            with self.subTest(msg=f"{t}.real-rhs"):
                b = np.random.rand(N)
                self.run_sparse_tests(self.solver(**self.kwargs), A, b)
            with self.subTest(msg=f"{t}.multi-real-rhs"):
                b = np.random.rand(N, 3)
                self.run_sparse_tests(self.solver(**self.kwargs), A, b)
            with self.subTest(msg=f"{t}.multi-real-rhs-lineardependent"):
                b = np.random.rand(N, 3)
                b[:, 1] = b[:, 2]
                self.run_sparse_tests(self.solver(**self.kwargs), A, b)

            if not pym.solvers.matrix_is_complex(A):
                continue

            with self.subTest(msg=f"{t}.complex-rhs"):
                b = np.random.rand(N) + 1j * np.random.rand(N)
                self.run_sparse_tests(self.solver(**self.kwargs), A, b)
            with self.subTest(msg=f"{t}.multi-complex-rhs"):
                b = np.random.rand(N, 3) + 1j * np.random.rand(N, 3)
                self.run_sparse_tests(self.solver(**self.kwargs), A, b)


class TestSparseDiagonal(GenericTestSolvers):
    solver = pym.solvers.SolverDiagonal
    matrices = [
        mat_real_diagonal,
        mat_real_spdiag,
        mat_complex_diagonal,
        mat_complex_spdiag,
    ]


class TestSparseLU(GenericTestSolvers):
    solver = pym.solvers.SolverSparseLU
    matrices = [
        mat_real_diagonal,
        mat_real_spdiag,
        mat_real_symm_pos_def,
        mat_real_symm_pos_def_dynamic,
        mat_real_symm_indef,
        mat_real_symm_indef_dynamic,
        mat_real_asymm,
        mat_real_symm_saddle,
        mat_complex_diagonal,
        mat_complex_spdiag,
        mat_complex_symm_pos_def_dynamic,
        mat_complex_symm_indef_dynamic,
        mat_complex_hermitian_1,
        mat_complex_hermitian_indef,
        mat_complex_hermitian_pos_def,
    ]


class TestCholeskyScikit(GenericTestSolvers):
    # I don't know why, but scikit is able to solve indefinite matrix as well. Maybe they do some LDL inside?
    solver = pym.solvers.SolverSparseCholeskyScikit
    matrices = [
        mat_real_diagonal,
        mat_real_spdiag,
        mat_real_symm_pos_def,
        mat_real_symm_pos_def_dynamic,
        mat_complex_hermitian_indef,
        mat_complex_hermitian_1,
        mat_complex_hermitian_pos_def,
    ]


class TestCholeskyCVXOPT(GenericTestSolvers):
    solver = pym.solvers.SolverSparseCholeskyCVXOPT
    matrices = [
        mat_real_diagonal,
        mat_real_spdiag,
        mat_real_symm_pos_def,
        mat_real_symm_pos_def_dynamic,
        mat_complex_hermitian_1,
        mat_complex_hermitian_pos_def,
    ]


class TestPardiso(GenericTestSolvers):
    solver = pym.solvers.SolverSparsePardiso
    matrices = [
        mat_real_diagonal,
        mat_real_spdiag,
        mat_real_symm_pos_def,
        mat_real_symm_pos_def_dynamic,
        mat_real_symm_indef,
        mat_real_symm_indef_dynamic,
        mat_real_symm_saddle,
        mat_real_asymm,
        # mat_complex_diagonal, # PyPardiso does not work for complex matrix yet
        # mat_complex_spdiag,
        # mat_complex_symm_pos_def_dynamic,
        # mat_complex_symm_indef_dynamic,
        # mat_complex_hermitian_1,
    ]


class TestSparseCG(GenericTestSolvers):
    solver = pym.solvers.CG
    # kwargs = dict(preconditioner=pym.solvers.ILU(), verbosity=1, tol=1e-9)
    kwargs = dict(preconditioner=pym.solvers.SOR(w=1.0), verbosity=1, tol=1e-9)
    # kwargs = dict(preconditioner=pym.solvers.DampedJacobi(), verbosity=1, tol=1e-9)
    matrices = [
        mat_real_diagonal,
        mat_real_spdiag,
        mat_real_symm_pos_def,
        mat_real_symm_pos_def_dynamic,
        # mat_real_symm_indef,
        # mat_real_symm_indef_dynamic,
        # mat_real_asymm,
        # mat_real_symm_saddle,
        mat_complex_diagonal,
        mat_complex_spdiag,
        # mat_complex_symm_pos_def_dynamic, # Works but needs many iterations
        # mat_complex_symm_indef_dynamic,
        mat_complex_hermitian_1,
        # mat_complex_hermitian_indef,
        mat_complex_hermitian_pos_def,
    ]


if __name__ == '__main__':
    unittest.main()
