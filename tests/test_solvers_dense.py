import unittest
import numpy as np
import numpy.testing as npt
import pymoto as pym
from pymoto.solvers import auto_determine_solver
import sys
import inspect
np.random.seed(0)


def make_symmetric(A):
    return (A + A.T)/2


def make_hermitian(A):
    return (A + A.conj().T)/2


def make_symm_pos_def(A):
    W, V = np.linalg.eig(make_symmetric(A))
    W[W < 0.1] += abs(min(W))+0.1
    return V@np.diag(W)@(V.T)


def make_herm_pos_def(A):
    W, V = np.linalg.eigh(make_hermitian(A))
    # Eigenvalues of hermitian matrix are always real
    W[W < 0.1] += abs(min(W)) + 0.1
    return V@np.diag(W)@(V.conj().T)


""" Perpare the set of matrices we want to test """
N = 10

# Real-valued matrices
mat_real_diagonal = np.diag(np.random.rand(N))
mat_real_symm = make_symmetric(np.random.rand(N, N))
mat_real_symm_pos_def = make_symm_pos_def(np.random.rand(N, N))
mat_real_asymm = np.random.rand(N, N)

# Complex-valued matrices
mat_complex_diagonal = np.diag(np.random.rand(N) + 1j*np.random.rand(N))
mat_complex_herm = make_hermitian(np.random.rand(N, N) + 1j*np.random.rand(N, N))
mat_complex_herm_pos_def = make_herm_pos_def(np.random.rand(N, N) + 1j*np.random.rand(N, N))
mat_complex_symm = make_symmetric(np.random.rand(N, N) + 1j*np.random.rand(N, N))
mat_complex_symm_pos_def = make_symm_pos_def(np.random.rand(N, N) + 1j*np.random.rand(N, N))
mat_complex_asymm = np.random.rand(N, N) + 1j*np.random.rand(N, N)


""" ------------------ TEST UTILITY FUNCTIONS -------------------- """


class TestGenericUtility(unittest.TestCase):
    """ Generic test class for testing functions on matrix properties that return True/False """
    data = []  # The data to test: list of tuples with (matrix, property)
    fn = None  # The test function to use, which return the tested property

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
            with self.subTest(msg=tags[i]):
                self.assertEqual(self.fn(A), res)


class TestIsComplex(TestGenericUtility):
    fn = staticmethod(pym.solvers.matrix_is_complex)
    data = [
        (mat_real_diagonal, False),
        (mat_real_symm, False),
        (mat_real_symm_pos_def, False),
        (mat_real_asymm, False),
        (mat_complex_diagonal, True),
        (mat_complex_herm, True),
        (mat_complex_herm_pos_def, True),
        (mat_complex_symm, True),
        (mat_complex_symm_pos_def, True),
        (mat_complex_asymm, True),
    ]


class TestIsDiagonal(TestGenericUtility):
    fn = staticmethod(pym.solvers.matrix_is_diagonal)
    data = [
        (mat_real_diagonal, True),
        (mat_real_symm, False),
        (mat_real_symm_pos_def, False),
        (mat_real_asymm, False),
        (mat_complex_diagonal, True),
        (mat_complex_herm, False),
        (mat_complex_herm_pos_def, False),
        (mat_complex_symm, False),
        (mat_complex_symm_pos_def, False),
        (mat_complex_asymm, False),
    ]


class TestIsSymmetric(TestGenericUtility):
    fn = staticmethod(pym.solvers.matrix_is_symmetric)
    data = [
        (mat_real_diagonal, True),
        (mat_real_symm, True),
        (mat_real_symm_pos_def, True),
        (mat_real_asymm, False),
        (mat_complex_diagonal, True),
        (mat_complex_herm, False),
        (mat_complex_herm_pos_def, False),
        (mat_complex_symm, True),
        (mat_complex_symm_pos_def, True),
        (mat_complex_asymm, False),
    ]


class TestIsHermitian(TestGenericUtility):
    fn = staticmethod(pym.solvers.matrix_is_hermitian)
    data = [
        (mat_real_diagonal, True),
        (mat_real_symm, True),
        (mat_real_symm_pos_def, True),
        (mat_real_asymm, False),
        (mat_complex_diagonal, False),
        (mat_complex_herm, True),
        (mat_complex_herm_pos_def, True),
        (mat_complex_symm, False),
        (mat_complex_symm_pos_def, False),
        (mat_complex_asymm, False),
    ]


class TestAutoSolver(TestGenericUtility):
    """ Check if the auto_determine returns correct solver types """
    fn = staticmethod(lambda A: type(auto_determine_solver(A)))
    data = [
        (mat_real_diagonal, pym.solvers.SolverDiagonal),
        (mat_real_symm, pym.solvers.SolverDenseCholesky),
        (mat_real_symm_pos_def, pym.solvers.SolverDenseCholesky),
        (mat_real_asymm, pym.solvers.SolverDenseLU),
        (mat_complex_diagonal, pym.solvers.SolverDiagonal),
        (mat_complex_herm, pym.solvers.SolverDenseCholesky),
        (mat_complex_herm_pos_def, pym.solvers.SolverDenseCholesky),
        (mat_complex_symm, pym.solvers.SolverDenseLDL),
        (mat_complex_symm_pos_def, pym.solvers.SolverDenseLDL),
        (mat_complex_asymm, pym.solvers.SolverDenseLU),
    ]


""" ------------------ TEST THE DENSE SOLVERS -------------------- """
# Solvers:
# - SolverDiagonal
# - SolverDenseQR
# - SolverDenseLU
# - SolverDenseLDL
# - SolverDenseCholesky


class GenericTestDenseSolvers(unittest.TestCase):
    """ Generic test runner for any dense solver and list of matrices """
    solver = None  # The solver to use
    matrices = []  # The list of matrices the solver should be able to handle

    @classmethod
    def setUpClass(cls) -> None:
        """ Skip test if the solver is not available """
        if cls.solver is None:
            raise unittest.SkipTest(f"No solver given for {cls.__name__}")
        if not cls.solver.defined:
            raise unittest.SkipTest(f"{type(cls.solver)} not available, lacking third-party libraries")

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
        atol = 1e-12
        # Get reference solutions
        A_N = A
        A_T = A.T
        A_H = A.conj().T
        xref_N = np.linalg.solve(A_N, b)
        xref_T = np.linalg.solve(A_T, b)
        xref_H = np.linalg.solve(A_H, b)

        # Do the solutions using provided solver
        solver.update(A)
        x_N = solver.solve(b, trans='N')
        x_T = solver.solve(b, trans='T')
        x_H = solver.solve(b, trans='H')

        # Check solution N
        npt.assert_allclose(x_N, xref_N)
        npt.assert_allclose(A_N @ x_N - b, 0.0, atol=atol)

        # Check transpose solution T
        npt.assert_allclose(x_T, xref_T)
        npt.assert_allclose(A_T @ x_T - b, 0.0, atol=atol)

        # Check adjoint solution H
        npt.assert_allclose(x_H, xref_H)
        npt.assert_allclose(A_H @ x_H - b, 0.0, atol=atol)

        # Run with LDAWrapper
        is_herm = pym.solvers.matrix_is_hermitian(A)
        is_symm = pym.solvers.matrix_is_symmetric(A)
        is_complex = np.iscomplexobj(b)
        LDAsolver = pym.solvers.LDAWrapper(solver)
        LDAsolver.update(A)

        # First solve (do the solution)
        x_N = LDAsolver.solve(b, trans='N')
        npt.assert_allclose(x_N, xref_N)
        npt.assert_allclose(A_N @ x_N - b, 0.0, atol=atol)
        self.assertTrue(np.all(LDAsolver._did_solve))  # the solution must have been done here

        # Second solve
        x_N = LDAsolver.solve(2.5*b, trans='N')
        npt.assert_allclose(x_N, 2.5*xref_N)
        npt.assert_allclose(A_N @ x_N - 2.5*b, 0.0, atol=atol)
        self.assertFalse(np.any(LDAsolver._did_solve))  # the solution is already done

        # Transpose solve
        x_T = LDAsolver.solve(3.8*b, trans='T')
        npt.assert_allclose(x_T, 3.8*xref_T)
        npt.assert_allclose(A_T @ x_T - 3.8*b, 0.0, atol=atol)
        if (is_herm and not is_complex) or is_symm:
            self.assertFalse(np.any(LDAsolver._did_solve))
        else:
            self.assertTrue(np.all(LDAsolver._did_solve))

        # Second transpose solve
        x_T = LDAsolver.solve(4.1*b, trans='T')
        npt.assert_allclose(x_T, 4.1*xref_T)
        npt.assert_allclose(A_T @ x_T - 4.1*b, 0.0, atol=atol)
        self.assertFalse(np.any(LDAsolver._did_solve))

        # Adjoint solve
        x_H = LDAsolver.solve(5.3*b, trans='H')
        npt.assert_allclose(x_H, 5.3*xref_H)
        npt.assert_allclose(A_H @ x_H - 5.3*b, 0.0, atol=atol)
        if (is_herm and not is_symm) or (is_symm and not is_complex) or not is_complex:
            self.assertFalse(np.any(LDAsolver._did_solve))
        else:
            self.assertTrue(np.all(LDAsolver._did_solve))

        # Second adjoint solve
        x_H = LDAsolver.solve(6.3 * b, trans='H')
        npt.assert_allclose(x_H, 6.3 * xref_H)
        npt.assert_allclose(A_H @ x_H - 6.3 * b, 0.0, atol=atol)
        self.assertFalse(np.any(LDAsolver._did_solve))

        # Normal solve with complex scaling
        LDAsolver = pym.solvers.LDAWrapper(solver)
        LDAsolver.update(A)
        x_N = LDAsolver.solve((6.3 + 1j*3.1) * b, trans='N')
        npt.assert_allclose(x_N, (6.3 + 1j*3.1) * xref_N)
        npt.assert_allclose(A_N @ x_N - (6.3 + 1j*3.1) * b, 0.0, atol=atol)
        self.assertTrue(np.all(LDAsolver._did_solve))

        # Normal solve with real scaling again
        x_N = LDAsolver.solve(0.9 * b, trans='N')
        npt.assert_allclose(x_N, 0.9 * xref_N)
        npt.assert_allclose(A_N @ x_N - 0.9 * b, 0.0, atol=atol)
        self.assertFalse(np.any(LDAsolver._did_solve))


    def test_all_matrices(self):
        """ Run the tests on all given matrices """
        tags = self.get_tags()
        for A, t in zip(self.matrices, tags):
            """ Test for different type of right-hand-sides """
            sys.stdout.write(f"Test \"{self.solver.__name__}\" for matrix \"{t}\"\n")
            N = A.shape[0]
            # with self.subTest(msg=f"{t}.real-rhs"):
            #     b = np.random.rand(N)
            #     self.run_solver(self.solver(), A, b)
            # with self.subTest(msg=f"{t}.complex-rhs"):
            #     b = np.random.rand(N) + 1j * np.random.rand(N)
            #     self.run_solver(self.solver(), A, b)
            with self.subTest(msg=f"{t}.multi-real-rhs"):
                b = np.random.rand(N, 3)
                self.run_solver(self.solver(), A, b)
            with self.subTest(msg=f"{t}.multi-complex-rhs"):
                b = np.random.rand(N, 3) + 1j * np.random.rand(N, 3)
                self.run_solver(self.solver(), A, b)


class TestDenseDiagonal(GenericTestDenseSolvers):
    solver = pym.solvers.SolverDiagonal
    matrices = [
        mat_real_diagonal,
        mat_complex_diagonal,
    ]


class TestDenseQR(GenericTestDenseSolvers):
    solver = pym.solvers.SolverDenseQR
    matrices = [
        mat_real_diagonal,
        mat_real_symm,
        mat_real_symm_pos_def,
        mat_real_asymm,
        mat_complex_diagonal,
        mat_complex_herm,
        mat_complex_herm_pos_def,
        mat_complex_symm,
        mat_complex_symm_pos_def,
        mat_complex_asymm,
    ]


class TestDenseLU(GenericTestDenseSolvers):
    solver = pym.solvers.SolverDenseLU
    matrices = [
        mat_real_diagonal,
        mat_real_symm,
        mat_real_symm_pos_def,
        mat_real_asymm,
        mat_complex_diagonal,
        mat_complex_herm,
        mat_complex_herm_pos_def,
        mat_complex_symm,
        mat_complex_symm_pos_def,
        mat_complex_asymm,
    ]


class TestDenseCholesky(GenericTestDenseSolvers):
    # The indefinite matrices use LDL instead as a backup
    solver = pym.solvers.SolverDenseCholesky
    matrices = [
        mat_real_diagonal,
        mat_real_symm,  # Uses backup solver
        mat_real_symm_pos_def,
        mat_complex_herm,  # Uses backup solver
        mat_complex_herm_pos_def,
    ]


class TestDenseLDL(GenericTestDenseSolvers):
    solver = pym.solvers.SolverDenseLDL
    matrices = [
        mat_real_diagonal,
        mat_real_symm,
        mat_real_symm_pos_def,
        mat_complex_diagonal,
        mat_complex_herm,
        mat_complex_herm_pos_def,
        mat_complex_symm,
        mat_complex_symm_pos_def,
    ]


""" ----------- TEST LINSOLVE MODULE ---------- """


class Symm(pym.Module):
    def _response(self, A):
        return (A + A.T)/2

    def _sensitivity(self, dB):
        return (dB + dB.T)/2


class Herm(pym.Module):
    def _response(self, A):
        return (A + A.conj().T) / 2

    def _sensitivity(self, dB):
        return (dB + dB.conj().T) / 2


class TestLinSolveModule_dense(unittest.TestCase):
    # The matrices to test for
    matrices = [
        mat_real_diagonal,
        mat_real_symm,
        mat_real_symm_pos_def,
        mat_real_asymm,
        mat_complex_diagonal,
        mat_complex_herm,
        mat_complex_herm_pos_def,
        mat_complex_symm,
        mat_complex_symm_pos_def,
        mat_complex_asymm,
    ]

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

    def run_solver(self, A, b):
        sA = pym.Signal("A", A)
        sb = pym.Signal("b", b)

        sx = pym.Signal("x")
        fn = pym.Network()
        symmetry = pym.solvers.matrix_is_symmetric(A)
        hermitian = pym.solvers.matrix_is_hermitian(A)

        if symmetry or hermitian:
            sAsys = pym.Signal("Asym", A)
            if symmetry:
                fn.append(Symm(sA, sAsys))
            elif hermitian:
                fn.append(Herm(sA, sAsys))
        else:
            sAsys = sA
        fn.append(pym.LinSolve([sAsys, sb], sx))

        fn.response()

        # Check residual
        self.assertTrue(np.allclose(sAsys.state@sx.state, sb.state))

        # Check finite difference
        # def tfn(x0, dx, df_an, df_fd): np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sA, sb], sx, test_fn=tfn, verbose=False)

    def test_all_matrices(self):
        """ Run the tests on all given matrices """
        tags = self.get_tags()
        for A, t in zip(self.matrices, tags):
            """ Test for different type of right-hand-sides """
            sys.stdout.write(f"Test LinSolve module for matrix \"{t}\"\n")
            N = A.shape[0]
            with self.subTest(msg=f"{t}.real-rhs"):
                b = np.random.rand(N)
                self.run_solver(A, b)
            with self.subTest(msg=f"{t}.complex-rhs"):
                b = np.random.rand(N) + 1j * np.random.rand(N)
                self.run_solver(A, b)
            with self.subTest(msg=f"{t}.multi-real-rhs"):
                b = np.random.rand(N, 3)
                self.run_solver(A, b)
            with self.subTest(msg=f"{t}.multi-complex-rhs"):
                b = np.random.rand(N, 3) + 1j * np.random.rand(N, 3)
                self.run_solver(A, b)


if __name__ == '__main__':
    unittest.main()
