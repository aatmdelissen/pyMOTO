import unittest
import numpy as np
import scipy as sp
from scipy.io import mmread
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
import scipy
import pymoto as pym
import sys
np.random.seed(0)


# TODO: Check hermitian and symmetry in complex matrices; check in which cases the adjoint system is solved
class Symm(pym.Module):
    def _response(self, A):
        return (A + A.T)/2
    def _sensitivity(self, dB):
        return (dB + dB.T)/2


class TestDenseSolvers(unittest.TestCase):
    def run_solver(self, solver, A, b):
        Aadj = A.conj().T
        xref = np.linalg.solve(A, b)
        xrefadj = np.linalg.solve(A.conj().T, b)
        solver.update(A)
        x = solver.solve(b)
        self.assertTrue(np.allclose(x, xref))
        self.assertTrue(np.allclose(A@x-b, 0.0))
        xadj = solver.adjoint(b)
        self.assertTrue(np.allclose(xadj, xrefadj))
        self.assertTrue(np.allclose(Aadj@xadj-b, 0.0))

    def test_real_diagonal(self):
        N = 10
        A = np.diag(np.random.rand(N)) # All positive
        b = np.random.rand(N)

        self.run_solver(pym.SolverDiagonal(), A, b)
        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseLDL(), A, b)
        self.run_solver(pym.SolverDenseCholesky(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDiagonal)

    def test_complex_diagonal(self):
        N = 10
        A = np.diag(np.random.rand(N) + 1j*np.random.rand(N))
        b = np.random.rand(N) + 1j*np.random.rand(N)

        self.run_solver(pym.SolverDiagonal(), A, b)
        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseLDL(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDiagonal)
        # Matrix is not hermitian, so no Cholesky

    def test_real(self):
        N = 10
        A = np.random.rand(N, N)
        b = np.random.rand(N)

        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDenseLU)

    def test_complex(self):
        N = 10
        A = np.random.rand(N, N) + 1j*np.random.rand(N, N)
        b = np.random.rand(N) + 1j*np.random.rand(N)

        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDenseLU)

    def test_real_symmetric(self):
        N = 10
        A = np.random.rand(N, N)
        A = (A + A.T)/2
        b = np.random.rand(N)

        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseLDL(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDenseCholesky)

    def test_complex_symmetric(self):
        N = 10
        A = np.random.rand(N, N) + 1j*np.random.rand(N, N)
        A = (A + A.T)/2
        b = np.random.rand(N) + 1j*np.random.rand(N)

        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseLDL(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDenseLDL)

    def test_complex_hermitian(self):
        N = 10
        A = np.random.rand(N, N) + 1j*np.random.rand(N, N)
        A = (A + A.conj().T)/2
        b = np.random.rand(N) + 1j*np.random.rand(N)

        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseLDL(), A, b)
        # self.run_solver(pym.SolverDenseCholesky(), A, b) # Skip because it is not necesarily positive definite
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDenseCholesky)

    def test_real_symmetric_positive_definite(self):
        N = 10
        Q = np.random.rand(N, N)
        A = Q@Q.T
        b = np.random.rand(N)
        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseLDL(), A, b)
        self.run_solver(pym.SolverDenseCholesky(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDenseCholesky)

    def test_complex_symmetric_positive_definite(self):
        N = 10
        Q = np.random.rand(N, N) + 1j*np.random.rand(N, N)
        A = Q@Q.T
        b = np.random.rand(N) + 1j*np.random.rand(N)
        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseLDL(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDenseLDL)

    def test_complex_hermitian_positive_definite(self):
        N = 10
        Q = np.random.rand(N, N) + 1j*np.random.rand(N, N)
        A = Q@Q.T.conj()
        A[np.arange(N), np.arange(N)] = np.real(A[np.arange(N), np.arange(N)])
        b = np.random.rand(N) + 1j*np.random.rand(N)
        self.run_solver(pym.SolverDenseQR(), A, b)
        self.run_solver(pym.SolverDenseLU(), A, b)
        self.run_solver(pym.SolverDenseLDL(), A, b)
        self.run_solver(pym.SolverDenseCholesky(), A, b)
        self.run_solver(pym.auto_determine_solver(A), A, b)
        self.assertIsInstance(pym.auto_determine_solver(A), pym.SolverDenseCholesky)


class TestLinSolveModule_dense(unittest.TestCase):

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
        """ Test asymmetric real dense matrix """
        N = 10
        A = np.random.rand(N, N)
        b = np.random.rand(N)
        self.use_solver(A, b)

    def test_asymmetric_real_multirhs(self):
        """ Test asymmetric real dense matrix with multiple rhs """
        N, M = 10, 3
        A = np.random.rand(N, N)
        b = np.random.rand(N, M)
        self.use_solver(A, b)

    def test_asymmetric_complex(self):
        """ Test asymmetric complex dense matrix """
        N = 10
        A = np.random.rand(N, N)+1j*np.random.rand(N, N)
        b = np.random.rand(N)+1j*np.random.rand(N)
        self.use_solver(A, b)

    def test_asymmetric_complex_multirhs(self):
        """ Test asymmetric complex dense matrix with multiple rhs """
        N, M = 10, 3
        A = np.random.rand(N, N)+1j*np.random.rand(N, N)
        b = np.random.rand(N, M)+1j*np.random.rand(N, M)
        self.use_solver(A, b)

    # # ------------- Symmetric -------------
    def test_symmetric_real(self):
        """ Test symmetric real dense matrix """
        N = 10
        A = np.random.rand(N, N)
        b = np.random.rand(N)
        self.use_solver(A, b, symmetry=True)

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


if __name__ == '__main__':
    unittest.main()  #TestSparseSolvers())
