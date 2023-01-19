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


np.random.seed(0)

# TODO Test the sparse solvers separately, and loop over the matrices
def load_matrix(filen):
    curdir = pathlib.Path(__file__).parent.resolve()
    return mmread(pathlib.Path(curdir, "data", filen))



def generate_real_diagonal(N):
    return spsp.spdiags(np.random.rand(N), 0, N, N).tocoo()  # All positive

def generate_complex_diagonal(N):
    return spsp.spdiags(np.random.rand(N) + 1j*np.random.rand(N), 0, N, N).tocoo()

def generate_symm_indef():
    A = load_matrix("zenios.mtx.gz")
    sigma = -0.1  # To prevent exact singularity
    As = A - sigma*spsp.eye(A.shape[0])
    # Calculate eigenvalues
    # solver = pym.SolverSparseLU().update(As)
    # opinv = spspla.LinearOperator(A.shape, matvec=solver.solve, rmatvec=solver.adjoint)
    # w, _ = spspla.eigsh(A, sigma=sigma, k=100, OPinv=opinv, ncv=200)]
    return As.tocoo()

def calculate_eigenvalues(A):
    return sp.linalg.eigvals(A.toarray())

def generate_dynamic(complex=False, indef=False):
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


# Perpare the matrices
matrix_real_spdiag = spsp.spdiags(np.random.rand(100), 0, 100, 100)
matrix_real_diagonal = generate_real_diagonal(100)
matrix_real_symm_pos_def = load_matrix("bcsstk14.mtx.gz")  # Symmetric positive definite
matrix_real_symm_pos_def_dynamic = generate_dynamic(complex=False, indef=False)
matrix_real_symm_indef = generate_symm_indef()  # Symmetric indefinite
matrix_real_symm_indef_dynamic = generate_dynamic(complex=False, indef=True)
matrix_real_asymm = load_matrix("impcol_a.mtx.gz")
matrix_complex_spdiag = matrix_real_spdiag + 1j*matrix_real_spdiag
matrix_complex_diagonal = generate_complex_diagonal(100)
matrix_complex_symm_pos_def_dynamic = generate_dynamic(complex=True, indef=False)
matrix_complex_symm_indef_dynamic = generate_dynamic(complex=True, indef=True)
matrix_complex_hermitian_1 = load_matrix("mhd1280b.mtx.gz")  #generate_hermitian(matrix_complex_symm_pos_def_dynamic)


class TestGenericUtility(unittest.TestCase):
    data = []
    results = []
    fn = None

    @classmethod
    def setUpClass(cls) -> None:
        if cls.fn is None:
            raise unittest.SkipTest(f"Skipping test {cls}")

    def get_tags(self):
        src = inspect.getsource(type(self))
        start = src.find('data')
        start += src[start:].find('=') + 1
        finish = start + src[start:].find(']')
        return [t.replace('\n','').replace(' ', '').replace('(', '').replace(',','').replace('True','').replace('False','') for t in src[start:finish].strip(' []').split(')')]

    def test_all_matrices(self):
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
            try:
                import cvxopt
                with self.subTest(msg=f"CVXOPT - {tags[i]}"):
                    # sys.stdout.write(f"Test {self.fn.__name__} for CVXOPT-matrix {tags[i]}\n")
                    if isinstance(A, spsp.coo_matrix):
                        Acvx = cvxopt.spmatrix(A.data, A.row.astype(int), A.col.astype(int))
                    self.assertEqual(self.fn(Acvx), res)
            except ImportError:
                sys.stdout.write("Skipping CVXOPT...\n")


class TestIsComplex(TestGenericUtility):
    # fn = staticmethod(pym.matrix_is_complex)
    data = [
        (matrix_real_diagonal, False),
        (matrix_real_spdiag, False),
        (matrix_real_symm_pos_def, False),
        (matrix_real_symm_pos_def_dynamic, False),
        (matrix_real_symm_indef, False),
        (matrix_real_symm_indef_dynamic, False),
        (matrix_real_asymm, False),
        (matrix_complex_diagonal, True),
        (matrix_complex_spdiag, True),
        (matrix_complex_symm_pos_def_dynamic, True),
        (matrix_complex_symm_indef_dynamic, True),
        (matrix_complex_hermitian_1, True),
    ]


class TestIsDiagonal(TestGenericUtility):
    # fn = staticmethod(pym.matrix_is_diagonal)
    data = [
        (matrix_real_diagonal, True),
        (matrix_real_spdiag, True),
        (matrix_real_symm_pos_def, False),
        (matrix_real_symm_pos_def_dynamic, False),
        (matrix_real_symm_indef, False),
        (matrix_real_symm_indef_dynamic, False),
        (matrix_real_asymm, False),
        (matrix_complex_diagonal, True),
        (matrix_complex_spdiag, True),
        (matrix_complex_symm_pos_def_dynamic, False),
        (matrix_complex_symm_indef_dynamic, False),
        (matrix_complex_hermitian_1, False),
    ]


class TestIsSymmetric(TestGenericUtility):
    # fn = staticmethod(pym.matrix_is_symmetric)
    data = [
        (matrix_real_diagonal, True),
        (matrix_real_spdiag, True),
        (matrix_real_symm_pos_def, True),
        (matrix_real_symm_pos_def_dynamic, True),
        (matrix_real_symm_indef, True),
        (matrix_real_symm_indef_dynamic, True),
        (matrix_real_asymm, False),
        (matrix_complex_diagonal, True),
        (matrix_complex_spdiag, True),
        (matrix_complex_symm_pos_def_dynamic, True),
        (matrix_complex_symm_indef_dynamic, True),
        (matrix_complex_hermitian_1, False),
    ]


class TestIsHermitian(TestGenericUtility):
    # fn = staticmethod(pym.matrix_is_hermitian)
    data = [
        (matrix_real_diagonal, True),
        (matrix_real_spdiag, True),
        (matrix_real_symm_pos_def, True),
        (matrix_real_symm_pos_def_dynamic, True),
        (matrix_real_symm_indef, True),
        (matrix_real_symm_indef_dynamic, True),
        (matrix_real_asymm, False),
        (matrix_complex_diagonal, False),
        (matrix_complex_spdiag, False),
        (matrix_complex_symm_pos_def_dynamic, False),
        (matrix_complex_symm_indef_dynamic, False),
        (matrix_complex_hermitian_1, True),
    ]


def run_solver(testcase, solver, A, b):
    solver.update(A)
    ref_solver = pym.SolverSparseLU(A)

    x = solver.solve(b)
    x_ref = ref_solver.solve(b)
    residual = A@x-b
    r = np.linalg.norm(residual) / np.linalg.norm(b)
    testcase.assertTrue(np.allclose(x, x_ref))
    testcase.assertTrue(r < 1e-4, msg=f"Residual is {r}")

    xadj = solver.adjoint(b)
    xadj_ref = ref_solver.adjoint(b)
    residual_adj = (A.conj().T)@xadj-b
    r_adj = np.linalg.norm(residual_adj) / np.linalg.norm(b)
    # testcase.assertTrue(np.allclose(xadj, xadj_ref))
    testcase.assertTrue(r_adj < 1e-4, msg=f"Adjoint residual is {r_adj} (normal residual = {r})")


def run_sparse(testcase, solver, Acoo, rhs):
    run_solver(testcase, solver, Acoo, rhs)
    run_solver(testcase, solver, Acoo.tocsr(), rhs)
    run_solver(testcase, solver, Acoo.tocsc(), rhs)


# SOLVERS:
# - scipy/umfpack lu
# - scikit cholmod
# - cvxopt cholmod
# - intel mkl pardiso

class GenericTestSolvers(unittest.TestCase):
    solver = None
    matrices = []

    @classmethod
    def setUpClass(cls) -> None:
        if (cls.solver is None or not cls.solver.defined):
            raise unittest.SkipTest(f"{type(cls.solver)} not available, lacking third-party libraries")

    def get_tags(self):
        src = inspect.getsource(type(self))
        start = src.find('matrices')
        start += src[start:].find('=') + 1
        finish = start + src[start:].find(']')
        return [t.strip(' ').replace('\n','').replace(' ', '') for t in src[start:finish].strip(' []').split(',')]

    def test_all_matrices(self):
        tags = self.get_tags()
        for A, t in zip(self.matrices, tags):
            with self.subTest(msg=t):
                sys.stdout.write(f"Test {self.solver} for {t}\n")
                b = np.random.rand(A.shape[0])
                if pym.matrix_is_complex(A):
                    b = b + 1j*np.random.rand(A.shape[0])
                run_sparse(self, self.solver(), A, b)


class TestSparseDiagonal(GenericTestSolvers):
    # solver = pym.SolverDiagonal
    matrices = [
        matrix_real_diagonal,
        matrix_real_spdiag,
        matrix_complex_diagonal,
        matrix_complex_spdiag,
    ]


class TestSparseLU(GenericTestSolvers):
    # solver = pym.SolverSparseLU
    matrices = [
        matrix_real_diagonal,
        matrix_real_spdiag,
        matrix_real_symm_pos_def,
        matrix_real_symm_pos_def_dynamic,
        matrix_real_symm_indef,
        matrix_real_symm_indef_dynamic,
        matrix_real_asymm,
        matrix_complex_diagonal,
        matrix_complex_spdiag,
        matrix_complex_symm_pos_def_dynamic,
        matrix_complex_symm_indef_dynamic,
        matrix_complex_hermitian_1,
    ]


class TestCholeskyScikit(GenericTestSolvers):
    # solver = pym.SolverSparseCholeskyScikit
    matrices = [
        matrix_real_diagonal,
        matrix_real_spdiag,
        matrix_real_symm_pos_def,
        matrix_real_symm_pos_def_dynamic,
        matrix_complex_diagonal,
        matrix_complex_spdiag,
        matrix_complex_hermitian_1,
        matrix_complex_symm_pos_def_dynamic,
    ]
    # The non-hermitian matrices are solved using A^-1 = A'(AA')^-1, which may result in low accuracy

class TestCholeskyCVXOPT(GenericTestSolvers):
    # solver = pym.SolverSparseCholeskyCVXOPT
    matrices = [
        matrix_real_diagonal,
        matrix_real_spdiag,
        matrix_real_symm_pos_def,
        matrix_real_symm_pos_def_dynamic,
        matrix_complex_hermitian_1,
    ]

class TestPardiso(GenericTestSolvers):
    solver = pym.SolverSparsePardiso
    matrices = [
        # matrix_real_diagonal,
        # matrix_real_spdiag,
        # matrix_real_symm_pos_def,
        # matrix_real_symm_pos_def_dynamic,
        # matrix_real_symm_indef,
        # matrix_real_symm_indef_dynamic,
        # matrix_real_asymm,
        # matrix_complex_diagonal,
        # matrix_complex_spdiag,
        # matrix_complex_symm_pos_def_dynamic,
        # matrix_complex_symm_indef_dynamic,
        matrix_complex_hermitian_1,
    ]
'''
class TestSparseSolvers(unittest.TestCase):
  

    def test_real_symmetric_indefinite(self):
        print("Real symmetric indefinite")
        matrix = "data/zenios.mtx.gz"
        sigma = -0.1
        A = mmread(matrix).tocsc()
        As = A - sigma*spsp.eye(A.shape[0])

        # Calculate eigenvalues
        # solver = pym.SolverSparsePardiso(mtype=-2)
        # solver.update(As)
        # opinv = spspla.LinearOperator(A.shape, matvec=solver.solve, rmatvec=solver.adjoint)
        # w, _ = spspla.eigsh(A, sigma=sigma, k=100, OPinv=opinv, ncv=200)

        b = np.random.rand(A.shape[0])
        self.run_sparse(pym.SolverSparsePardiso(mtype=1), As, b)
        self.run_sparse(pym.SolverSparsePardiso(mtype=-2), As, b)
        # self.run_sparse(pym.SolverSparsePardiso(mtype=2), As, b)
        # self.run_sparse(pym.SolverSparsePardiso(mtype=3))
        # if pym.SolverSparseCholeskyCVXOPT.defined: # TODO Somehow doesn't work??
        #     self.run_sparse(pym.SolverSparseCholeskyCVXOPT(), As, b)
        if pym.SolverSparseCholeskyScikit.defined:
            self.run_sparse(pym.SolverSparseCholeskyScikit(), As, b)
        self.run_sparse(pym.SolverSparseLU(), As, b)

    def test_real_symmetric_shifted(self):
        print("Real symmetric shifted")
        k_matrix = "data/bcsstk26.mtx.gz"
        m_matrix = "data/bcsstm26.mtx.gz"

        K = mmread(k_matrix).tocsc()
        M = mmread(m_matrix).tocsc()

        b = np.random.rand(K.shape[0])

        solver = pym.SolverSparsePardiso(mtype=2)
        solver.update(K)
        opinv = spspla.LinearOperator(K.shape, matvec=solver.solve, rmatvec=solver.adjoint)
        w, _ = spspla.eigsh(K, M=M, sigma=0, k=100, OPinv=opinv, ncv=200)
        eigfreqs = np.sqrt(w)

        freq = (eigfreqs[0] + eigfreqs[1])/2
        Ks = K - freq**2 * M

        self.run_sparse(pym.SolverSparsePardiso(mtype=-2), Ks, b)
        # if pym.SolverSparseCholeskyCVXOPT.defined:
        #     self.run_sparse(pym.SolverSparseCholeskyCVXOPT(), Ks, b)
        if pym.SolverSparseCholeskyScikit.defined:
            self.run_sparse(pym.SolverSparseCholeskyScikit(), Ks, b)
        self.run_sparse(pym.SolverSparseLU(), Ks, b)



class TestLinSolveModule_sparse(unittest.TestCase):
    def use_solver(self, A, b, symmetry=False):
        sA = pym.Signal("A", A)
        sb = pym.Signal("b", b)

        sx = pym.Signal("x")
        fn = pym.Network()
        if symmetry:
            sAsys = pym.Signal("Asym", A)
            fn.append(Symm(sA, sAsys))
        else:
            sAsys = sA
        fn.append(pym.LinSolve([sAsys, sb], sx))

        fn.response()

        # Check residual
        self.assertTrue(np.allclose(sAsys.state@sx.state, sb.state))

        # Check finite difference
        test_fn = lambda x0, dx, df_an, df_fd: self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        # test_fn = lambda x0, dx, df_an, df_fd:np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
        pym.finite_difference(fn, [sA, sb], sx, test_fn=test_fn, verbose=False)

    
    # ------------- Asymmetric -------------
    def test_asymmetric_real(self):
        """ Test asymmetric real sparse matrix """
        N = 10
        A = np.random.rand(N, N)
        b = np.random.rand(N)
        self.use_solver(A, b)

    def test_asymmetric_real_multirhs(self):
        """ Test asymmetric real sparse matrix with multiple rhs """
        N, M = 10, 3
        A = np.random.rand(N, N)
        b = np.random.rand(N, M)
        self.use_solver(A, b)

    def test_asymmetric_complex(self):
        """ Test asymmetric complex sparse matrix """
        N = 10
        A = np.random.rand(N, N)+1j*np.random.rand(N, N)
        b = np.random.rand(N)+1j*np.random.rand(N)
        self.use_solver(A, b)

    def test_asymmetric_complex_multirhs(self):
        """ Test asymmetric complex sparse matrix with multiple rhs """
        N, M = 10, 3
        A = np.random.rand(N, N)+1j*np.random.rand(N, N)
        b = np.random.rand(N, M)+1j*np.random.rand(N, M)
        self.use_solver(A, b)
    
    # # ------------- Symmetric -------------
    def test_symmetric_real_compliance2d(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        N = 10 # Number of elements
        dom = pym.DomainDefinition(N, N)
        sx = pym.Signal('x', np.random.rand(dom.nel))
        fixed_nodes = dom.get_nodenumber(0, np.arange(0, N+1))
        bc = np.concatenate((fixed_nodes*2, fixed_nodes*2+1))
        iforce = dom.get_nodenumber(N, np.arange(0, N+1))*2 + 1
        sf = pym.Signal('f', np.zeros(dom.nnodes*2))
        sf.state[iforce] = 1.0

        fn = pym.Network()
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), dom, bc=bc))
        su = fn.append(pym.LinSolve([sK, sf], pym.Signal('u')))

        fn.response()

        self.assertTrue(np.allclose(sK.state@su.state, sf.state)) # Check residual
        # Check finite difference
        test_fn = lambda x0, dx, df_an, df_fd: self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sf], su, test_fn=test_fn, dx=1e-5, tol=1e-4, verbose=False)

    def test_symmetric_real_compliance3d(self):
        """ Test symmetric real sparse matrix (compliance in 3D)"""
        N = 3 # Number of elements
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

        self.assertTrue(np.allclose(sK.state@su.state, sf.state)) # Check residual
        # Check finite difference
        test_fn = lambda x0, dx, df_an, df_fd: self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sf], su, test_fn=test_fn, dx=1e-5, tol=1e-4, verbose=False)

    def test_symmetric_complex_dyncompliance2d(self):
        """ Test symmetric complex sparse matrix (dynamic compliance in 3D)"""
        N = 10 # Number of elements
        dom = pym.DomainDefinition(N, N)
        sx = pym.Signal('x', np.random.rand(dom.nel))
        fixed_nodes = dom.get_nodenumber(0, np.arange(0, N+1))
        bc = np.concatenate((fixed_nodes*2, fixed_nodes*2+1))
        iforce = dom.get_nodenumber(N, np.arange(0, N+1))*2 + 1
        sf = pym.Signal('f', np.zeros(dom.nnodes*2))
        sf.state[iforce] = 1.0

        class DynamicMatrix(pym.Module):
            alpha = 0.05
            beta = 0.05
            def _response(self, K, M, omega):
                return K + 1j * omega * (self.alpha * M + self.beta * K) - omega**2 * M
            def _sensitivity(self, dZ):
                K, M, omega = [s.state for s in self.sig_in]
                dK = np.real(dZ) + omega*self.beta*np.imag(dZ)
                dM = -omega**2 * dZ.real + omega*self.alpha*np.imag(dZ)
                dZrM = dZ.real.contract(M)
                dZiK = dZ.imag.contract(K)
                dZiM = dZ.imag.contract(K)
                domega = self.alpha * dZiM + self.beta * dZiK - 2*omega*dZrM
                return dK, dM, domega

        sOmega = pym.Signal('omega', 1.34)
        fn = pym.Network()
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), dom, bc=bc))
        sM = fn.append(pym.AssembleMass(sx, pym.Signal('M'), dom, bc=bc))
        sZ = fn.append(DynamicMatrix([sK, sM, sOmega], pym.Signal('Z')))

        su = fn.append(pym.LinSolve([sZ, sf], pym.Signal('u')))#, solver=pym.SolverSparseCholeskyCVXOPT()))

        fn.response()  # TODO Pardiso does not support complex matrix yet

        self.assertTrue(np.allclose(sZ.state@su.state, sf.state)) # Check residual
        # Check finite difference
        test_fn = lambda x0, dx, df_an, df_fd: self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sf, sOmega], su, test_fn=test_fn, dx=1e-5, tol=1e-4, verbose=False)

    
    def test_symmetric_real_multirhs(self):
        """ Test symmetric real dense matrix with multiple rhs """
        N, M = 10, 3
        A = np.random.rand(N, N)
        b = np.random.rand(N, M)
        self.use_solver(A, b, symmetry=True)

    def test_symmetric_complex(self):
        """ Test symmetric complex dense matrix """
        N = 10
        A = np.random.rand(N, N)+1j*np.random.rand(N, N)
        b = np.random.rand(N)+1j*np.random.rand(N)
        self.use_solver(A, b, symmetry=True)

    def test_symmetric_complex_multirhs(self):
        """ Test symmetric complex dense matrix with multiple rhs """
        N, M = 10, 3
        A = np.random.rand(N, N)+1j*np.random.rand(N, N)
        b = np.random.rand(N, M)+1j*np.random.rand(N, M)
        self.use_solver(A, b, symmetry=True)
    '''


if __name__ == '__main__':
    unittest.main()#TestSparseSolvers())