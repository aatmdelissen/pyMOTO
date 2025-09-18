import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym


class Symm(pym.Module):
    def __call__(self, A):
        return (A + A.T)/2

    def _sensitivity(self, dB):
        return (dB + dB.T)/2


class Herm(pym.Module):
    def __call__(self, A):
        return (A + A.T.conj())/2

    def _sensitivity(self, dB):
        return (dB + dB.T.conj())/2


def make_random(N: int) -> np.ndarray:
    return np.random.rand(N, N)


def make_positive_definite(A: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(A)  # Eigenvectors
    D = np.diag(np.random.rand(*A.diagonal().shape) + 0.3)  # Positive eigenvalues only
    return Q @ D @ Q.T.conj()  # Return hermitian


np.random.seed(0)


@pytest.mark.parametrize('modifier', [None, 'symmetric', 'hermitian'])
@pytest.mark.parametrize('A', [make_random(10), make_random(10) + 1j * make_random(10)], ids=['real', 'complex'])
def test_dense_eigensolver(A, modifier):
    sA = pym.Signal("A", A)

    if modifier is not None:
        Modifier = Herm() if 'hermitian' in modifier.lower() else Symm()
        sAsys = Modifier(sA)
        sAsys.tag = f"A{modifier}"
    else:
        sAsys = sA
    slambda, sU = pym.EigenSolve()(sAsys)
    slambda.tag = 'lambda'
    sU.tag = 'eigvec'

    sLamsum = pym.EinSum('i->')(slambda)
    sLamsum.tag = "lambdasum"
    sVecsum = pym.EinSum('ij->')(sU)
    sVecsum.tag = "vecsum"
    sSumsum = pym.MathGeneral('inp0 + inp1')(sLamsum, sVecsum)
    sSumsum.tag = "allsum"

    # Test residual and normalization
    npt.assert_allclose(sAsys.state @ sU.state, sU.state @ np.diag(slambda.state))
    npt.assert_allclose(np.diag(sU.state.T @ sU.state), 1.0)

    # Check finite difference (test_fn with quite loose tolerances because of numerical precision)
    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
    pym.finite_difference(sA, [slambda, sU, sLamsum, sVecsum, sSumsum], test_fn=tfn, verbose=True)


def test_eig_vs_eigh():
    import scipy.linalg as spla
    import timeit
    N = 50
    A = make_random(N)
    Q, _ = np.linalg.qr(np.random.rand(N, N))
    D = np.diag(np.random.rand(N) * 0.9 + 0.1)
    B = Q @ D @ Q.T
    
    def solve_eig(A=A, B=B):
        return spla.eig(A, B)

    def solve_eigh(A=A, B=B):
        return spla.eigh(A, B)
    print(f"Eig = {timeit.timeit(solve_eig, number=1000)}s")
    print(f"Eigh = {timeit.timeit(solve_eigh, number=1000)}s")


@pytest.mark.parametrize('Bmodifier', [None, 'symmetric', 'hermitian', 'hermitian_positive_definite'])
@pytest.mark.parametrize('B', [make_random(10), make_random(10) + 1j * make_random(10)], ids=['Breal', 'Bcomplex'])
@pytest.mark.parametrize('Amodifier', [None, 'symmetric', 'hermitian'])
@pytest.mark.parametrize('A', [make_random(10), make_random(10) + 1j * make_random(10)], ids=['Areal', 'Acomplex'])
def test_dense_generalized_eigensolver(A, B, Amodifier, Bmodifier):
    if Bmodifier is not None and 'positive_definite' in Bmodifier.lower():
        B = make_positive_definite(B)
    sA = pym.Signal("A", A)
    sB = pym.Signal("B", B)

    if Amodifier is not None:
        AMod = Herm if 'hermitian' in Amodifier.lower() else Symm
        sAsys = AMod()(sA)
        sAsys.tag = f'A_{Amodifier}'
    else:
        sAsys = sA

    if Bmodifier is not None:
        BMod = Herm if 'hermitian' in Bmodifier.lower() else Symm
        sBsys = BMod()(sB)
        sBsys.tag = f'B_{Bmodifier}'
    else:
        sBsys = sB

    slambda, sU = pym.EigenSolve()(sAsys, sBsys)
    slambda.tag = 'lambda'
    sU.tag = 'eigvec'

    sLamsum = pym.EinSum('i->')(slambda)
    sLamsum.tag = "lambdasum"
    sVecsum = pym.EinSum('ij->')(sU)
    sVecsum.tag = "vecsum"
    sSumsum = pym.MathGeneral('inp0 + inp1')(sLamsum, sVecsum)
    sSumsum.tag = "allsum"

    print(f"lambdas = {slambda.state}")

    # Test residual and normalization
    npt.assert_allclose(sAsys.state @ sU.state, sBsys.state @ sU.state @ np.diag(slambda.state))
    npt.assert_allclose(np.diag(sU.state.T @ sBsys.state @ sU.state), 1.0)

    # Check finite difference (test_fn with quite loose tolerances because of numerical precision)
    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
    pym.finite_difference([sA, sB], [sLamsum, sVecsum, sSumsum], test_fn=tfn, verbose=True)

    # def test_asymmetric_real(self):
    #     """ Test asymmetric real dense matrix """
    #     N = 10
    #     A = np.random.rand(N, N)
    #     # Matrix B must be positive definite, so we construct it using eigenmodes Q and (positive) eigenvalues D
    #     Q, _ = np.linalg.qr(np.random.rand(N, N))
    #     D = np.diag(np.random.rand(N) + 0.3)
    #     self.use_solver(A, Q@D@Q.T)
    #
    # def test_asymmetric_complex(self):
    #     """ Test asymmetric complex dense matrix """
    #     N = 10
    #     A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    #     Q, _ = np.linalg.qr(np.random.rand(N, N) + 1j * np.random.rand(N, N))
    #     D = np.diag(np.random.rand(N) + 0.3)
    #     self.use_solver(A, Q@D@Q.T.conj())
    #
    # # # ------------- Symmetric -------------
    # def test_symmetric_real(self):
    #     """ Test symmetric real dense matrix """
    #     N = 10
    #     A = np.random.rand(N, N)
    #     Q, _ = np.linalg.qr(np.random.rand(N, N))
    #     D = np.diag(np.random.rand(N) * 0.9 + 0.1)
    #     self.use_solver(A, Q @ D @ Q.T, symmetry=True)
    #
    # def test_symmetric_complex(self):
    #     """ Test symmetric complex dense matrix """
    #     N = 10
    #     A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    #     Q, _ = np.linalg.qr(np.random.rand(N, N) + 1j * np.random.rand(N, N))
    #     D = np.diag(np.random.rand(N) * 0.9 + 0.1)
    #     self.use_solver(A, Q@D@Q.T.conj(), symmetry=True)
    #
    # def test_hermitian_complex(self):
    #     """ Test symmetric complex dense matrix """
    #     N = 10
    #     A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    #     Q, _ = np.linalg.qr(np.random.rand(N, N) + 1j * np.random.rand(N, N))
    #     D = np.diag(np.random.rand(N) * 0.9 + 0.1)
    #     self.use_solver(A, Q@D@Q.T.conj(), hermitian=True)


@pytest.mark.parametrize("generalized",
                         [pytest.param(True, id='generalized'),
                          pytest.param(False, id='normal')])
def test_eigensolve_sparse(generalized):
    np.random.seed(0)
    nx, ny = 3, 6
    domain = pym.DomainDefinition(nx, ny)
    bc = (domain.nodes[0, :]*2 + np.arange(2)[None]).flatten()
    s_x = pym.Signal('x', state=np.ones(domain.nel)*0.5)

    s_K = pym.AssembleStiffness(domain=domain, bc=bc)(s_x)
    input_signals = [s_K, ]
    if generalized:
        s_M = pym.AssembleMass(domain=domain, bc=bc, ndof=domain.dim)(s_x)
        input_signals.append(s_M)
    s_lam, s_V = pym.EigenSolve()(*input_signals)
    s_lam.tag, s_V.tag = 'lam', 'V'

    if generalized:
        npt.assert_allclose(s_K.state @ s_V.state, s_M.state @ s_V.state @ np.diag(s_lam.state), atol=1e-10)
    else:
        npt.assert_allclose(s_K.state @ s_V.state, s_V.state @ np.diag(s_lam.state), atol=1e-10)

    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-3)
    # Test eigenvalue sensitivities
    pym.finite_difference(s_x, s_lam, test_fn=tfn, verbose=True, dx=1e-6)
    # Test eigenvector sensitivities
    for i in range(6):
        pym.finite_difference(s_x, s_V[:, i], test_fn=tfn, verbose=True, dx=1e-6)
    pym.finite_difference(s_x, s_V, test_fn=tfn, verbose=True, dx=1e-6)


@pytest.mark.parametrize("complex_matrix", [True, False])
def test_eigensolve_sparse_generalized(complex_matrix):
    np.random.seed(0)
    nx, ny = 3, 4
    domain = pym.DomainDefinition(nx, ny)
    bc = (domain.nodes[0, :]*2 + np.arange(2)[None]).flatten()
    xvec = np.ones(domain.nel)*0.5
    s_x = pym.Signal('x', state=xvec)

    s_K = pym.AssembleStiffness(domain=domain, bc=bc, e_modulus=1. + 1j)(s_x)
    s_M = pym.AssembleMass(domain=domain, bc=bc, ndof=domain.dim)(s_x)
    s_lam, s_V = pym.EigenSolve()(s_K, s_M)
    s_lam.tag, s_V.tag = 'lam', 'V'

    npt.assert_allclose(s_K.state @ s_V.state, s_M.state @ s_V.state @ np.diag(s_lam.state), atol=1e-10)

    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-3)

    pym.finite_difference(s_x, s_lam, dx=1e-6, test_fn=tfn, verbose=True)
    for i in range(6):
        pym.finite_difference(s_x, s_V[:, i], dx=1e-6, test_fn=tfn, verbose=True)
    pym.finite_difference(s_x, s_V, dx=1e-6, test_fn=tfn, verbose=True)


if __name__ == '__main__':
    pytest.main([__file__])
