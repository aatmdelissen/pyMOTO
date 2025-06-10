import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym
from pymoto.solvers import auto_determine_solver
# from dataclasses import dataclass
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
    return V @ np.diag(W) @ V.T


def make_herm_pos_def(A):
    W, V = np.linalg.eigh(make_hermitian(A))
    # Eigenvalues of hermitian matrix are always real
    W[W < 0.1] += abs(min(W)) + 0.1
    return V @ np.diag(W) @ V.conj().T


""" Perpare the set of matrices we want to test """
N = 10

# @dataclass
# class MatrixData:
#     matrix: np.ndarray
#     is_complex: bool
#     is_diagonal: bool
#     is_symmetric: bool
#     is_hermitian: bool
#     auto_solver_type: pym.solvers.LinearSolver

# Real-valued matrices
all_matrices = dict(
    mat_real_diagonal=np.diag(np.random.rand(N)),
    mat_real_symm=make_symmetric(np.random.rand(N, N)),
    mat_real_symm_pos_def=make_symm_pos_def(np.random.rand(N, N)),
    mat_real_asymm=np.random.rand(N, N),
    # Complex-valued matrices
    mat_complex_diagonal=np.diag(np.random.rand(N) + 1j*np.random.rand(N)),
    mat_complex_herm=make_hermitian(np.random.rand(N, N) + 1j*np.random.rand(N, N)),
    mat_complex_herm_pos_def=make_herm_pos_def(np.random.rand(N, N) + 1j*np.random.rand(N, N)),
    mat_complex_symm=make_symmetric(np.random.rand(N, N) + 1j*np.random.rand(N, N)),
    mat_complex_symm_pos_def=make_symm_pos_def(np.random.rand(N, N) + 1j*np.random.rand(N, N)),
    mat_complex_asymm=np.random.rand(N, N) + 1j*np.random.rand(N, N),
)


""" ------------------ TEST UTILITY FUNCTIONS -------------------- """
@pytest.mark.parametrize('Atag, expected', [
        ('mat_real_diagonal', False),
        ('mat_real_symm', False),
        ('mat_real_symm_pos_def', False),
        ('mat_real_asymm', False),
        ('mat_complex_diagonal', True),
        ('mat_complex_herm', True),
        ('mat_complex_herm_pos_def', True),
        ('mat_complex_symm', True),
        ('mat_complex_symm_pos_def', True),
        ('mat_complex_asymm', True),
    ])
def test_matrix_is_complex(Atag, expected):
    A = all_matrices[Atag]
    assert pym.solvers.matrix_is_complex(A) == expected


@pytest.mark.parametrize('Atag, expected', [
        ('mat_real_diagonal', True),
        ('mat_real_symm', False),
        ('mat_real_symm_pos_def', False),
        ('mat_real_asymm', False),
        ('mat_complex_diagonal', True),
        ('mat_complex_herm', False),
        ('mat_complex_herm_pos_def', False),
        ('mat_complex_symm', False),
        ('mat_complex_symm_pos_def', False),
        ('mat_complex_asymm', False),
    ])
def test_matrix_is_diagonal(Atag, expected):
    A = all_matrices[Atag]
    assert pym.solvers.matrix_is_diagonal(A) == expected


@pytest.mark.parametrize('Atag, expected', [
        ('mat_real_diagonal', True),
        ('mat_real_symm', True),
        ('mat_real_symm_pos_def', True),
        ('mat_real_asymm', False),
        ('mat_complex_diagonal', True),
        ('mat_complex_herm', False),
        ('mat_complex_herm_pos_def', False),
        ('mat_complex_symm', True),
        ('mat_complex_symm_pos_def', True),
        ('mat_complex_asymm', False),
    ])
def test_matrix_is_symmetric(Atag, expected):
    A = all_matrices[Atag]
    assert pym.solvers.matrix_is_symmetric(A) == expected


@pytest.mark.parametrize('Atag, expected', [
        ('mat_real_diagonal', True),
        ('mat_real_symm', True),
        ('mat_real_symm_pos_def', True),
        ('mat_real_asymm', False),
        ('mat_complex_diagonal', False),
        ('mat_complex_herm', True),
        ('mat_complex_herm_pos_def', True),
        ('mat_complex_symm', False),
        ('mat_complex_symm_pos_def', False),
        ('mat_complex_asymm', False),
])
def test_matrix_is_hermitian(Atag, expected):
    A = all_matrices[Atag]
    assert pym.solvers.matrix_is_hermitian(A) == expected


@pytest.mark.parametrize('Atag, expected', [
        ('mat_real_diagonal', pym.solvers.SolverDiagonal),
        ('mat_real_symm', pym.solvers.SolverDenseCholesky),
        ('mat_real_symm_pos_def', pym.solvers.SolverDenseCholesky),
        ('mat_real_asymm', pym.solvers.SolverDenseLU),
        ('mat_complex_diagonal', pym.solvers.SolverDiagonal),
        ('mat_complex_herm', pym.solvers.SolverDenseCholesky),
        ('mat_complex_herm_pos_def', pym.solvers.SolverDenseCholesky),
        ('mat_complex_symm', pym.solvers.SolverDenseLDL),
        ('mat_complex_symm_pos_def', pym.solvers.SolverDenseLDL),
        ('mat_complex_asymm', pym.solvers.SolverDenseLU),
])
def test_correct_auto_solver(Atag, expected):
    A = all_matrices[Atag]
    assert isinstance(auto_determine_solver(A), expected)


""" ------------------ TEST THE DENSE SOLVERS -------------------- """
# Solvers:
# - SolverDiagonal
# - SolverDenseQR
# - SolverDenseLU
# - SolverDenseLDL
# - SolverDenseCholesky


def run_solver_test(solver_type, A, b):
    solver = solver_type()
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
    is_diag = pym.solvers.matrix_is_diagonal(A)
    is_conjubable_rhs = (np.real(b).max() == np.real(b).min() == 0) or (np.imag(b).max() == np.imag(b).min() == 0)
    LDAsolver = pym.solvers.LDAWrapper(solver)
    LDAsolver.update(A)

    # ----- Normal solve
    x_N = LDAsolver.solve(b, trans='N')  # Solve Ax = b
    npt.assert_allclose(x_N, xref_N)
    npt.assert_allclose(A_N @ x_N - b, 0.0, atol=atol)
    if is_diag:
        assert not any(LDAsolver._did_solve)  # Diagonal matrix is solved directly
    else:
        assert all(LDAsolver._did_solve)  # the solution must have been done here

    # Second normal solve, scaled with the first
    x_N = LDAsolver.solve(2.5 * b, trans='N')  # Solve Ax = 2.5*b (no solve required)
    npt.assert_allclose(x_N, 2.5 * xref_N)
    npt.assert_allclose(A_N @ x_N - 2.5 * b, 0.0, atol=atol)
    assert not any(LDAsolver._did_solve)  # the solution is already done

    # ----- Transpose solve
    x_T = LDAsolver.solve(3.8 * b, trans='T')
    npt.assert_allclose(x_T, 3.8 * xref_T)
    npt.assert_allclose(A_T @ x_T - 3.8 * b, 0.0, atol=atol)
    if is_diag or is_symm or (is_herm and is_conjubable_rhs):
        assert not any(LDAsolver._did_solve)
    else:
        assert all(LDAsolver._did_solve)

    # Second transpose solve, scaled with the first
    x_T = LDAsolver.solve(4.1 * b, trans='T')
    npt.assert_allclose(x_T, 4.1 * xref_T)
    npt.assert_allclose(A_T @ x_T - 4.1 * b, 0.0, atol=atol)
    assert not any(LDAsolver._did_solve)

    # ----- Conjugate transpose solve
    x_H = LDAsolver.solve(5.3 * b, trans='H')
    npt.assert_allclose(x_H, 5.3 * xref_H)
    npt.assert_allclose(A_H @ x_H - 5.3 * b, 0.0, atol=atol)

    if is_diag:
        assert not any(LDAsolver._did_solve)  # Diagonal matrix is solved directly
    elif is_conjubable_rhs:  # Real rhs: same as 'T', Hermitian matrix, same as 'N'
        assert not any(LDAsolver._did_solve)
    elif is_herm and np.iscomplexobj(A):
        assert not any(LDAsolver._did_solve)
    else:
        assert all(LDAsolver._did_solve)  # Only solve for complex non-hermitian matrix

    # Second conjugate transpose solve
    x_H = LDAsolver.solve(6.3 * b, trans='H')
    npt.assert_allclose(x_H, 6.3 * xref_H)
    npt.assert_allclose(A_H @ x_H - 6.3 * b, 0.0, atol=atol)
    assert not any(LDAsolver._did_solve)

    # REINITIALIZE LDAS
    # Normal solve with complex scaling
    LDAsolver = pym.solvers.LDAWrapper(solver)
    LDAsolver.update(A)
    x_N = LDAsolver.solve((6.3 + 1j * 3.1) * b, trans='N')
    npt.assert_allclose(x_N, (6.3 + 1j * 3.1) * xref_N)
    npt.assert_allclose(A_N @ x_N - (6.3 + 1j * 3.1) * b, 0.0, atol=atol)
    if is_diag:
        assert not any(LDAsolver._did_solve)  # Diagonal matrix is solved directly
    else:
        assert all(LDAsolver._did_solve)

    # Normal solve with real scaling again
    x_N = LDAsolver.solve(0.9 * b, trans='N')
    npt.assert_allclose(x_N, 0.9 * xref_N)
    npt.assert_allclose(A_N @ x_N - 0.9 * b, 0.0, atol=atol)
    assert not any(LDAsolver._did_solve)


def construct_b(b_type: str, b_shape: int, n: int):
    if b_shape is None:
        b = np.random.rand(n)
    else:
        b = np.random.rand(n, b_shape)
    if 'complex' in b_type:
        b = b + 1j * np.random.rand(*b.shape)
    elif 'imaginary' in b_type:
        b = 1j * b
    return b


@pytest.mark.parametrize('b_shape', [None, 1, 3], ids=['singleRHS', 'columnRHS', 'multiRHS'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', ['mat_real_diagonal', 'mat_complex_diagonal'])
def test_diagonal(Atag, b_type, b_shape):
    A = all_matrices[Atag]
    b = construct_b(b_type, b_shape, A.shape[0])
    run_solver_test(pym.solvers.SolverDiagonal, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3], ids=['singleRHS', 'columnRHS', 'multiRHS'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', all_matrices.keys())
def test_qr(Atag, b_type, b_shape):
    A = all_matrices[Atag]
    b = construct_b(b_type, b_shape, A.shape[0])
    run_solver_test(pym.solvers.SolverDenseQR, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3], ids=['singleRHS', 'columnRHS', 'multiRHS'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', all_matrices.keys())
def test_lu(Atag, b_type, b_shape):
    A = all_matrices[Atag]
    b = construct_b(b_type, b_shape, A.shape[0])
    run_solver_test(pym.solvers.SolverDenseLU, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3], ids=['singleRHS', 'columnRHS', 'multiRHS'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', [
        'mat_real_diagonal',
        'mat_real_symm',  # Uses backup solver
        'mat_real_symm_pos_def',
        'mat_complex_herm',  # Uses backup solver
        'mat_complex_herm_pos_def',
])
def test_cholesky(Atag, b_type, b_shape):
    # The indefinite matrices use LDL instead as a backup
    A = all_matrices[Atag]
    b = construct_b(b_type, b_shape, A.shape[0])
    run_solver_test(pym.solvers.SolverDenseCholesky, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3], ids=['singleRHS', 'columnRHS', 'multiRHS'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', [
        'mat_real_diagonal',
        'mat_real_symm',
        'mat_real_symm_pos_def',
        'mat_complex_diagonal',
        'mat_complex_herm',
        'mat_complex_herm_pos_def',
        'mat_complex_symm',
        'mat_complex_symm_pos_def',
])
def test_ldl(Atag, b_type, b_shape):
    A = all_matrices[Atag]
    b = construct_b(b_type, b_shape, A.shape[0])
    run_solver_test(pym.solvers.SolverDenseLDL, A, b)


""" ----------- TEST LINSOLVE MODULE ---------- """


class Symm(pym.Module):
    def __call__(self, A):
        return (A + A.T)/2

    def _sensitivity(self, dB):
        return (dB + dB.T)/2


class Herm(pym.Module):
    def __call__(self, A):
        return (A + A.conj().T) / 2

    def _sensitivity(self, dB):
        return (dB + dB.conj().T) / 2


@pytest.mark.parametrize('b_shape', [None, 1, 3], ids=['singleRHS', 'columnRHS', 'multiRHS'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', all_matrices.keys())
def test_linsolve_module(Atag, b_type, b_shape):
    A = all_matrices[Atag]
    b = construct_b(b_type, b_shape, A.shape[0])
    sA = pym.Signal("A", A)
    sb = pym.Signal("b", b)

    with pym.Network() as fn:
        if pym.solvers.matrix_is_symmetric(A):
            sAsys = Symm()(sA)
            sAsys.tag = 'Asymm'
        elif pym.solvers.matrix_is_hermitian(A):
            sAsys = Herm()(sA)
            sAsys.tag = 'Aherm'
        else:
            sAsys = sA

        sx = pym.LinSolve()(sAsys, sb)
        sx.tag = 'x'

    # Check residual
    npt.assert_allclose(sAsys.state @ sx.state, sb.state)

    # Check finite difference
    def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
    pym.finite_difference([sA, sb], sx, function=fn, test_fn=tfn, verbose=False)


if __name__ == '__main__':
    pytest.main()
