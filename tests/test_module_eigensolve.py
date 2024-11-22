import unittest
import numpy as np
import numpy.testing as npt
import pymoto as pym
np.random.seed(0)


class Symm(pym.Module):
    def _response(self, A):
        return (A + A.T)/2

    def _sensitivity(self, dB):
        return (dB + dB.T)/2


class Herm(pym.Module):
    def _response(self, A):
        return (A + A.T.conj())/2

    def _sensitivity(self, dB):
        return (dB + dB.T.conj())/2


class TestEigenSolverDense(unittest.TestCase):
    def use_solver(self, A, symmetry=False, hermitian=False):
        sA = pym.Signal("A", A)

        slambda = pym.Signal("lambda")
        sU = pym.Signal("eigvec")

        fn = pym.Network()
        if symmetry or hermitian:
            if hermitian and symmetry and np.iscomplexobj(A):
                raise RuntimeError("Complex matrix cannot be symmetric and hermitian at the same time")
            Modifier = Herm if hermitian else Symm
            sAsys = pym.Signal("Asym", A)
            fn.append(Modifier(sA, sAsys))
        else:
            sAsys = sA
        fn.append(pym.EigenSolve(sAsys, [slambda, sU]))
        sLamsum, sVecsum = pym.Signal("lambdasum"), pym.Signal("vecsum")
        fn.append(pym.EinSum(slambda, sLamsum, expression='i->'))
        fn.append(pym.EinSum(sU, sVecsum, expression='ij->'))
        sSumsum = pym.Signal("allsum")
        fn.append(pym.MathGeneral([sLamsum, sVecsum], sSumsum, expression='inp0 + inp1'))
        fn.response()

        # Test residual and normalization
        self.assertTrue(np.allclose(sAsys.state @ sU.state, sU.state @ np.diag(slambda.state)))
        self.assertTrue(np.allclose(np.diag(sU.state.T @ sU.state), 1.0))

        # Check finite difference (test_fn with quite loose tolerances because of numerical precision)
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, sA, [slambda, sU, sLamsum, sVecsum, sSumsum], test_fn=tfn, verbose=True)

    def test_asymmetric_real(self):
        """ Test asymmetric real dense matrix """
        N = 10
        A = np.random.rand(N, N)
        self.use_solver(A)

    def test_asymmetric_complex(self):
        """ Test asymmetric complex dense matrix """
        N = 10
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        self.use_solver(A)

    # # ------------- Symmetric -------------
    def test_symmetric_real(self):
        """ Test symmetric real dense matrix """
        N = 10
        A = np.random.rand(N, N)
        self.use_solver(A, symmetry=True)

    def test_symmetric_complex(self):
        """ Test symmetric complex dense matrix """
        N = 10
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        self.use_solver(A, symmetry=True)

    def test_hermitian_complex(self):
        """ Test symmetric complex dense matrix """
        N = 10
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        self.use_solver(A, hermitian=True)


class TestEigenSolverDense_Generalized(unittest.TestCase):
    def use_solver(self, A, B, symmetry=False, hermitian=False):
        sA = pym.Signal("A", A)
        sB = pym.Signal("B", B)

        slambda = pym.Signal("lambda")
        sU = pym.Signal("eigvec")

        fn = pym.Network()
        if symmetry or hermitian:
            if hermitian and symmetry and np.iscomplexobj(A):
                raise RuntimeError("Complex matrix cannot be symmetric and hermitian at the same time")
            Modifier = Herm if hermitian else Symm
            sAsys = pym.Signal("Asym", A)
            fn.append(Modifier(sA, sAsys))

            sBsys = pym.Signal("Bsym", B)
            fn.append(Modifier(sB, sBsys))
        else:
            sAsys = sA
            sBsys = sB
        fn.append(pym.EigenSolve([sAsys, sBsys], [slambda, sU]))
        sLamsum, sVecsum = pym.Signal("lambdasum"), pym.Signal("vecsum")
        fn.append(pym.EinSum(slambda, sLamsum, expression='i->'))
        fn.append(pym.EinSum(sU, sVecsum, expression='ij->'))
        sSumsum = pym.Signal("allsum")
        fn.append(pym.MathGeneral([sLamsum, sVecsum], sSumsum, expression='inp0 + inp1'))
        fn.response()

        print(f"lambdas = {slambda.state}")

        # Test residual and normalization
        self.assertTrue(np.allclose(sAsys.state @ sU.state, sBsys.state @ sU.state @ np.diag(slambda.state)))
        self.assertTrue(np.allclose(np.diag(sU.state.T @ sBsys.state @ sU.state), 1.0))

        # Check finite difference (test_fn with quite loose tolerances because of numerical precision)
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sA, sB], [sLamsum, sVecsum, sSumsum], test_fn=tfn, verbose=True)

    def test_asymmetric_real(self):
        """ Test asymmetric real dense matrix """
        N = 10
        A = np.random.rand(N, N)
        # Matrix B must be positive definite, so we construct it using eigenmodes Q and (positive) eigenvalues D
        Q, _ = np.linalg.qr(np.random.rand(N, N))
        D = np.diag(np.random.rand(N) + 0.3)
        self.use_solver(A, Q@D@Q.T)

    def test_asymmetric_complex(self):
        """ Test asymmetric complex dense matrix """
        N = 10
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        Q, _ = np.linalg.qr(np.random.rand(N, N) + 1j * np.random.rand(N, N))
        D = np.diag(np.random.rand(N) + 0.3)
        self.use_solver(A, Q@D@Q.T.conj())

    # # ------------- Symmetric -------------
    def test_symmetric_real(self):
        """ Test symmetric real dense matrix """
        N = 10
        A = np.random.rand(N, N)
        Q, _ = np.linalg.qr(np.random.rand(N, N))
        D = np.diag(np.random.rand(N) * 0.9 + 0.1)
        self.use_solver(A, Q @ D @ Q.T, symmetry=True)

    def test_symmetric_complex(self):
        """ Test symmetric complex dense matrix """
        N = 10
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        Q, _ = np.linalg.qr(np.random.rand(N, N) + 1j * np.random.rand(N, N))
        D = np.diag(np.random.rand(N) * 0.9 + 0.1)
        self.use_solver(A, Q@D@Q.T.conj(), symmetry=True)

    def test_hermitian_complex(self):
        """ Test symmetric complex dense matrix """
        N = 10
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        Q, _ = np.linalg.qr(np.random.rand(N, N) + 1j * np.random.rand(N, N))
        D = np.diag(np.random.rand(N) * 0.9 + 0.1)
        self.use_solver(A, Q@D@Q.T.conj(), hermitian=True)


def test_eigenvalue_sparse():
    np.random.seed(0)
    nx, ny = 3, 6
    domain = pym.DomainDefinition(nx, ny)
    bc = (domain.nodes[0, :]*2 + np.arange(2)[None]).flatten()
    s_x = pym.Signal('x', state=np.ones(domain.nel)*0.5)

    fn = pym.Network()
    s_K = fn.append(pym.AssembleStiffness(s_x, domain=domain, bc=bc))
    s_lam, s_V = fn.append(pym.EigenSolve(s_K))

    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-5)
    pym.finite_difference(fn, s_x, s_lam, test_fn=tfn, verbose=True)


def test_eigenvalue_sparse_generalized():
    np.random.seed(0)
    nx, ny = 3, 6
    domain = pym.DomainDefinition(nx, ny)
    bc = (domain.nodes[0, :]*2 + np.arange(2)[None]).flatten()
    s_x = pym.Signal('x', state=np.ones(domain.nel)*0.5)

    fn = pym.Network()
    s_K = fn.append(pym.AssembleStiffness(s_x, domain=domain, bc=bc))
    s_M = fn.append(pym.AssembleMass(s_x, domain=domain, bc=bc, ndof=domain.dim))
    s_lam, s_V = fn.append(pym.EigenSolve([s_K, s_M]))

    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-5)
    pym.finite_difference(fn, s_x, s_lam, test_fn=tfn, verbose=True)


def test_eigenvector_sparse():
    np.random.seed(0)
    nx, ny = 3, 6
    domain = pym.DomainDefinition(nx, ny)
    bc = (domain.nodes[0, :]*2 + np.arange(2)[None]).flatten()
    s_x = pym.Signal('x', state=np.ones(domain.nel)*0.5)

    fn = pym.Network()
    s_K = fn.append(pym.AssembleStiffness(s_x, domain=domain, bc=bc))
    s_lam, s_V = fn.append(pym.EigenSolve(s_K))

    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-4)
    for i in range(6):
        pym.finite_difference(fn, s_x, s_V[:, i], test_fn=None, verbose=True)
    pym.finite_difference(fn, s_x, s_V, test_fn=tfn, verbose=True)


def test_eigenvector_sparse_generalized():
    np.random.seed(0)
    nx, ny = 3, 6
    domain = pym.DomainDefinition(nx, ny)
    bc = (domain.nodes[0, :]*2 + np.arange(2)[None]).flatten()
    s_x = pym.Signal('x', state=np.ones(domain.nel)*0.5)

    fn = pym.Network()
    s_K = fn.append(pym.AssembleStiffness(s_x, domain=domain, bc=bc))
    s_M = fn.append(pym.AssembleMass(s_x, domain=domain, bc=bc, ndof=domain.dim))
    s_lam, s_V = fn.append(pym.EigenSolve([s_K, s_M]))

    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-4)
    for i in range(6):
        pym.finite_difference(fn, s_x, s_V[:, i], test_fn=tfn, verbose=True)
    pym.finite_difference(fn, s_x, s_V, test_fn=tfn, verbose=True)


if __name__ == '__main__':
    unittest.main()
