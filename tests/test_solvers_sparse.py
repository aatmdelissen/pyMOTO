import unittest
import numpy as np
import scipy as sp
from scipy.io import mmread  # For importing files
import pathlib  # For importing files
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
import scipy
import pymoto as pym
import sys
import inspect

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
    # solver = pym.SolverSparseLU().update(As)
    # opinv = spspla.LinearOperator(A.shape, matvec=solver.solve, rmatvec=solver.adjoint)
    # w, _ = spspla.eigsh(A, sigma=sigma, k=100, OPinv=opinv, ncv=200)]
    return As.tocoo()


def generate_dynamic(complex=False, indef=False):
    """ Generate a dynamic stiffness matrix, either complex or not, and indefinite or not"""
    K = load_matrix("bcsstk26.mtx.gz")
    M = load_matrix("bcsstm26.mtx.gz")
    solver = pym.SolverSparseLU().update(K.tocsc())
    opinv = spspla.LinearOperator(K.shape, matvec=solver.solve, rmatvec=solver.adjoint)
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
    solver = pym.SolverSparseLU().update(K.tocsc())
    opinv = spspla.LinearOperator(K.shape, matvec=solver.solve, rmatvec=solver.adjoint)
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
    fn = staticmethod(pym.matrix_is_complex)
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
    fn = staticmethod(pym.matrix_is_diagonal)
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
    fn = staticmethod(pym.matrix_is_symmetric)
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
    fn = staticmethod(pym.matrix_is_hermitian)
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
        return [t.strip(' ').replace('\n', '').replace(' ', '') for t in src[start:finish].strip(' []').split(',')]

    def run_solver(self, solver, A, b):
        """ Run the actual test for given solver, matrix and right-hand-side """
        # Reference solution
        x_ref = pym.SolverSparseLU(A).solve(b)
        xadj_ref = pym.SolverSparseLU(A.conj().T).solve(b)

        # Calculate the solution and the adjoint solution
        solver.update(A)
        x = solver.solve(b)
        xadj = solver.adjoint(b)

        # Check residual and solution
        residual = A @ x - b
        r = np.linalg.norm(residual) / np.linalg.norm(b)
        self.assertTrue(r < 1e-4, msg=f"Residual is {r}")
        self.assertTrue(np.allclose(x, x_ref))

        # Check adjoint residual and solution
        residual_adj = (A.conj().T) @ xadj - b
        r_adj = np.linalg.norm(residual_adj) / np.linalg.norm(b)
        self.assertTrue(r_adj < 1e-4, msg=f"Adjoint residual is {r_adj} (normal residual = {r})")
        self.assertTrue(np.allclose(xadj, xadj_ref))

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
                self.run_sparse_tests(self.solver(), A, b)
            with self.subTest(msg=f"{t}.multi-real-rhs"):
                b = np.random.rand(N, 3)
                self.run_sparse_tests(self.solver(), A, b)

            if not pym.matrix_is_complex(A):
                continue

            with self.subTest(msg=f"{t}.complex-rhs"):
                b = np.random.rand(N) + 1j * np.random.rand(N)
                self.run_sparse_tests(self.solver(), A, b)
            with self.subTest(msg=f"{t}.multi-complex-rhs"):
                b = np.random.rand(N, 3) + 1j * np.random.rand(N, 3)
                self.run_sparse_tests(self.solver(), A, b)


class TestSparseDiagonal(GenericTestSolvers):
    solver = pym.SolverDiagonal
    matrices = [
        mat_real_diagonal,
        mat_real_spdiag,
        mat_complex_diagonal,
        mat_complex_spdiag,
    ]


class TestSparseLU(GenericTestSolvers):
    solver = pym.SolverSparseLU
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
    solver = pym.SolverSparseCholeskyScikit
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
    solver = pym.SolverSparseCholeskyCVXOPT
    matrices = [
        mat_real_diagonal,
        mat_real_spdiag,
        mat_real_symm_pos_def,
        mat_real_symm_pos_def_dynamic,
        mat_complex_hermitian_1,
        mat_complex_hermitian_pos_def,
    ]


class TestPardiso(GenericTestSolvers):
    solver = pym.SolverSparsePardiso
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


""" ------------------ TEST LINSOLVE MODULE -------------------- """


class DynamicMatrix(pym.Module):
    alpha = 0.5
    beta = 0.5

    def _response(self, K, M, omega):
        return K + 1j * omega * (self.alpha * M + self.beta * K) - omega ** 2 * M

    def _sensitivity(self, dZ):
        K, M, omega = [s.state for s in self.sig_in]
        dK = np.real(dZ) - (omega * self.beta) * np.imag(dZ)
        dM = (-omega ** 2) * np.real(dZ) - (omega * self.alpha) * np.imag(dZ)
        dZrM = np.real(dZ).contract(M)
        dZiK = np.imag(dZ).contract(K)
        dZiM = np.imag(dZ).contract(M)
        domega = -self.beta * dZiK - self.alpha * dZiM - 2 * omega * dZrM
        return dK, dM, domega

class TestLinSolveModule_sparse(unittest.TestCase):
    # # ------------- Symmetric -------------
    def test_symmetric_real_compliance2d(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        N = 10  # Number of elements
        dom = pym.DomainDefinition(N, N)
        sx = pym.Signal('x', np.random.rand(dom.nel))
        fixed_nodes = dom.get_nodenumber(0, np.arange(0, N+1))
        bc = np.concatenate((fixed_nodes*2, fixed_nodes*2+1))
        # Setup different rhs types
        iforce_x = dom.get_nodenumber(N, np.arange(0, N + 1)) * 2  # Force in x-direction
        iforce_y = dom.get_nodenumber(N, np.arange(0, N + 1)) * 2 + 1  # Force in y-direction

        force_vecs = dict()

        # Single force
        f = np.zeros(dom.nnodes*2)
        f[iforce_x] = 1.0
        force_vecs['single_real'] = f

        # Multiple rhs
        f = np.zeros((dom.nnodes * 2, 2))
        f[iforce_x, 0] = 1.0
        f[iforce_y, 1] = 1.0
        force_vecs['multiple_real'] = f

        for k, f in force_vecs.items():
            with self.subTest(f"RHS-{k}"):
                sf = pym.Signal('f', f)

                fn = pym.Network()
                sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), dom, bc=bc))
                su = fn.append(pym.LinSolve([sK, sf], pym.Signal('u')))

                fn.response()

                self.assertTrue(np.allclose(sK.state@su.state, sf.state))  # Check residual
                # Check finite difference
                # def tfn(x0, dx, df_an, df_fd): np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
                def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
                pym.finite_difference(fn, [sx, sf], su, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_symmetric_real_compliance3d(self):
        """ Test symmetric real sparse matrix (compliance in 3D)"""
        N = 3  # Number of elements
        dom = pym.DomainDefinition(N, N, N)
        sx = pym.Signal('x', np.random.rand(dom.nel))
        jfix, kfix = np.meshgrid(np.arange(0, N+1), np.arange(0, N+1), indexing='ij')
        fixed_nodes = dom.get_nodenumber(0, jfix, kfix).flatten()
        bc = np.concatenate((fixed_nodes*3, fixed_nodes*3+1, fixed_nodes*3+2))
        iforce = dom.get_nodenumber(N, np.arange(0, N+1), np.arange(0, N+1))*3 + 1
        sf = pym.Signal('f', np.zeros(dom.nnodes*3))
        sf.state[iforce] = 1.0

        fn = pym.Network()
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), dom, bc=bc))
        su = fn.append(pym.LinSolve([sK, sf], pym.Signal('u')))

        fn.response()

        self.assertTrue(np.allclose(sK.state@su.state, sf.state))  # Check residual
        # Check finite difference
        # def tfn(x0, dx, df_an, df_fd): np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=2e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sf], su, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_symmetric_complex_dyncompliance2d(self):
        """ Test symmetric complex sparse matrix (dynamic compliance in 2D)"""
        N = 5  # Number of elements
        dom = pym.DomainDefinition(N, N)
        sx = pym.Signal('x', np.random.rand(dom.nel))
        fixed_nodes = dom.get_nodenumber(0, np.arange(0, N+1))
        bc = np.concatenate((fixed_nodes*2, fixed_nodes*2+1))
        iforce = dom.get_nodenumber(N, np.arange(0, N+1))*2 + 1
        sf = pym.Signal('f', np.zeros(dom.nnodes*2))
        sf.state[iforce] = 1.0

        sOmega = pym.Signal('omega', 0.1)
        fn = pym.Network()
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), dom, bc=bc))
        sM = fn.append(pym.AssembleMass(sx, pym.Signal('M'), dom, bc=bc))
        sZ = fn.append(DynamicMatrix([sK, sM, sOmega], pym.Signal('Z')))

        su = fn.append(pym.LinSolve([sZ, sf], pym.Signal('u')))

        fn.response()

        # spspla.eigsh(sK.state, M=sM.state, k=6, sigma=0.0)

        self.assertTrue(np.allclose(sZ.state@su.state, sf.state))  # Check residual
        # Check finite difference
        # def tfn(x0, dx, df_an, df_fd): np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sf, sOmega], su, test_fn=tfn, dx=1e-7, tol=1e-4, verbose=False)


class TestSystemOfEquations(unittest.TestCase):
    def test_sparse_symmetric_real_compliance2d_single_load(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        N=10
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_horizontal = dofs_left[0::2]
        dofs_left_vertical = dofs_left[1::2]

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Setup solution vectors and rhs
        ff = np.zeros_like(free_dofs, dtype=float)
        ff[:] = np.random.rand(len(free_dofs))
        u = np.zeros_like(all_dofs, dtype=float)

        u[dofs_left_vertical] = np.random.rand(len(dofs_left_vertical))
        up = u[prescribed_dofs]

        sff = pym.Signal('ff', ff)
        sup = pym.Signal('up', up)

        fn = pym.Network()
        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), domain))
        su = fn.append(pym.SystemOfEquations([sK, sff, sup], free=free_dofs, prescribed=prescribed_dofs))
        sc = fn.append(pym.EinSum([su[0], su[1]], expression='i,i->'))
        fn.response()
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sff, sup], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_sparse_symmetric_real_compliance2d_single_multi_load(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        N=10
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_horizontal = dofs_left[0::2]
        dofs_left_vertical = dofs_left[1::2]

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Setup solution vectors and rhs
        ff = np.zeros((len(free_dofs), 2), dtype=float)
        ff[:, :] = np.random.rand(np.shape(ff)[0], np.shape(ff)[1])
        u = np.zeros((len(all_dofs), 2), dtype=float)

        u[dofs_left_vertical, 0] = np.random.rand(len(dofs_left_vertical))
        u[dofs_left_horizontal, 1] = np.random.rand(len(dofs_left_vertical))
        up = u[prescribed_dofs, :]

        sff = pym.Signal('ff', ff)
        sup = pym.Signal('up', up)

        fn = pym.Network()
        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), domain))
        su = fn.append(pym.SystemOfEquations([sK, sff, sup], prescribed=prescribed_dofs))
        sc1 = fn.append(pym.EinSum([su[0][:, 0], su[1][:, 0]], expression='i,i->'))
        sc2 = fn.append(pym.EinSum([su[0][:, 1], su[1][:, 1]], expression='i,i->'))
        sc = fn.append(pym.MathGeneral([sc1, sc2], expression='inp0 + inp1'))
        fn.response()
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sff, sup], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)


if __name__ == '__main__':
    unittest.main()
