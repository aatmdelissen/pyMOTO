import inspect
import pathlib  # For importing files
import sys
import pytest

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


def generate_diagonal(N):
    """ Generates a complex diagonal matrix of size N """
    return spsp.spdiags(np.random.rand(N), 0, N, N)


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
mat_1280b = load_matrix("mhd1280b.mtx.gz")

all_matrices = dict(
    # Real-valued matrices
    mat_real_spdiag=generate_diagonal(100),  # Diagonal (spdiag format)
    mat_real_diagonal=generate_diagonal(100).tocoo(),  # Diagonal (coo format)
    mat_real_symm_pos_def=load_matrix("bcsstk14.mtx.gz"),  # Symmetric positive definite
    mat_real_symm_pos_def_dynamic=generate_dynamic(complex=False, indef=False),  # Symmetric positive definite
    mat_real_symm_indef=generate_symm_indef(),  # Symmetric indefinite
    mat_real_symm_indef_dynamic=generate_dynamic(complex=False, indef=True),  # Symmetric indefinite
    mat_real_symm_saddle=generate_saddlepoint(),  # Indefinite saddlepoinnt
    mat_real_asymm=load_matrix("impcol_a.mtx.gz"),  # Asymmetric
    # Complex-valued matrices
    mat_complex_spdiag=generate_diagonal(100) + 1j*generate_diagonal(100),  # Diagonal (spdiag format)
    mat_complex_diagonal=(generate_diagonal(100) + 1j*generate_diagonal(100)).tocoo(),  # Diagonal (coo format)
    mat_complex_symm_pos_def_dynamic=generate_dynamic(complex=True, indef=False),  # Symmetric positive definite
    mat_complex_symm_indef_dynamic=generate_dynamic(complex=True, indef=True),  # Symmetric indefinite
    mat_complex_hermitian_1=mat_1280b,  # Hermitian
    mat_complex_hermitian_indef=mat_1280b - spsp.eye(*mat_1280b.shape),
    mat_complex_hermitian_pos_def=mat_1280b + spsp.eye(*mat_1280b.shape),
)

""" ------------------ TEST UTILITY FUNCTIONS -------------------- """


mat_type_converters = [lambda A: A.toarray(), lambda A: A, lambda A: A.tocsr(), lambda A: A.tocsc()]
mat_type_ids = ['dense', 'coo', 'csr', 'csc']
if _has_cvxopt:
    mat_type_converters.append(lambda A: cvxopt.spmatrix(A.data, A.row.astype(int), A.col.astype(int)) if isinstance(A, spsp.coo_matrix) else A)  # TODO Fix for diagonal matrix
    mat_type_ids.append('cvx')


@pytest.mark.parametrize('mat_converter', mat_type_converters, ids=mat_type_ids)
@pytest.mark.parametrize('Atag, expected', [
        ('mat_real_diagonal', False),
        ('mat_real_spdiag', False),
        ('mat_real_symm_pos_def', False),
        ('mat_real_symm_pos_def_dynamic', False),
        ('mat_real_symm_indef', False),
        ('mat_real_symm_indef_dynamic', False),
        ('mat_real_asymm', False),
        ('mat_complex_diagonal', True),
        ('mat_complex_spdiag', True),
        ('mat_complex_symm_pos_def_dynamic', True),
        ('mat_complex_symm_indef_dynamic', True),
        ('mat_complex_hermitian_1', True),
        ('mat_complex_hermitian_indef', True),
        ('mat_complex_hermitian_pos_def', True),
])
def test_is_complex(Atag, expected, mat_converter):
    assert pym.solvers.matrix_is_complex(mat_converter(all_matrices[Atag])) == expected


@pytest.mark.parametrize('mat_converter', mat_type_converters, ids=mat_type_ids)
@pytest.mark.parametrize('Atag, expected', [
    ('mat_real_diagonal', True),
    ('mat_real_spdiag', True),
    ('mat_real_symm_pos_def', False),
    ('mat_real_symm_pos_def_dynamic', False),
    ('mat_real_symm_indef', False),
    ('mat_real_symm_indef_dynamic', False),
    ('mat_real_asymm', False),
    ('mat_complex_diagonal', True),
    ('mat_complex_spdiag', True),
    ('mat_complex_symm_pos_def_dynamic', False),
    ('mat_complex_symm_indef_dynamic', False),
    ('mat_complex_hermitian_1', False),
    ('mat_complex_hermitian_indef', False),
    ('mat_complex_hermitian_pos_def', False),
])
def test_is_diagonal(Atag, expected, mat_converter):
    assert pym.solvers.matrix_is_diagonal(mat_converter(all_matrices[Atag])) == expected


@pytest.mark.parametrize('mat_converter', mat_type_converters, ids=mat_type_ids)
@pytest.mark.parametrize('Atag, expected', [
    ('mat_real_diagonal', True),
    ('mat_real_spdiag', True),
    ('mat_real_symm_pos_def', True),
    ('mat_real_symm_pos_def_dynamic', True),
    ('mat_real_symm_indef', True),
    ('mat_real_symm_indef_dynamic', True),
    ('mat_real_asymm', False),
    ('mat_complex_diagonal', True),
    ('mat_complex_spdiag', True),
    ('mat_complex_symm_pos_def_dynamic', True),
    ('mat_complex_symm_indef_dynamic', True),
    ('mat_complex_hermitian_1', False),
    ('mat_complex_hermitian_indef', False),
    ('mat_complex_hermitian_pos_def', False),
])
def test_is_symmetric(Atag, expected, mat_converter):
    assert pym.solvers.matrix_is_symmetric(mat_converter(all_matrices[Atag])) == expected


@pytest.mark.parametrize('mat_converter', mat_type_converters, ids=mat_type_ids)
@pytest.mark.parametrize('Atag, expected', [
    ('mat_real_diagonal', True),
    ('mat_real_spdiag', True),
    ('mat_real_symm_pos_def', True),
    ('mat_real_symm_pos_def_dynamic', True),
    ('mat_real_symm_indef', True),
    ('mat_real_symm_indef_dynamic', True),
    ('mat_real_asymm', False),
    ('mat_complex_diagonal', False),
    ('mat_complex_spdiag', False),
    ('mat_complex_symm_pos_def_dynamic', False),
    ('mat_complex_symm_indef_dynamic', False),
    ('mat_complex_hermitian_1', True),
    ('mat_complex_hermitian_indef', True),
    ('mat_complex_hermitian_pos_def', True),
])
def test_is_hermitian(Atag, expected, mat_converter):
    assert pym.solvers.matrix_is_hermitian(mat_converter(all_matrices[Atag])) == expected



""" ------------------ TEST THE SPARSE SOLVERS -------------------- """
# SOLVERS:
# - scipy/umfpack lu
# - scikit cholmod
# - cvxopt cholmod
# - intel mkl pardiso


def run_solver_test(solver_type, A, b, **kwargs):
    """ Run the actual test for given solver, matrix and right-hand-side """
    if not solver_type.defined:
        pytest.skip(f"Solver {solver_type} is not defined")
    solver = solver_type(**kwargs)

    # Store matrix versions
    mats = dict()
    mats['T'] = A.T
    mats['N'] = A
    mats['H'] = A.conj().T

    # Update solver
    solver.update(A)

    for k, Ai in mats.items():
        print(f'Run mode = {k}')
        xref = pym.solvers.SolverSparseLU(Ai).solve(b)

        # Calculate the solution and the adjoint solution
        x = solver.solve(b, trans=k)

        # Check residual and solution
        r = b - Ai @ x
        tval = np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=0)
        assert tval.max() < 1e-4

        npt.assert_allclose(x, xref, atol=1e-10, rtol=1e-4)


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

# For CVXOPT matrices, these behave wierd and are not tested. With these matrices you are on your own


@pytest.mark.parametrize('b_shape', [None, 1, 3, 4], ids=['singleRHS', 'columnRHS', 'multiRHS', 'multiRHS_lindep'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', ['mat_real_diagonal', 'mat_real_spdiag', 'mat_complex_diagonal', 'mat_complex_spdiag'])
@pytest.mark.parametrize('mat_converter', mat_type_converters[:4], ids=mat_type_ids[:4])
def test_sparse_diagonal(Atag, b_type, b_shape, mat_converter):
    A = mat_converter(all_matrices[Atag])
    b = construct_b(b_type, b_shape, A.shape[0])
    if not np.iscomplexobj(A) and np.iscomplexobj(b):
        pytest.skip("complex b with real A")  # TODO Allow complex rhs for real matrix
    if b_shape == 4:  # Create a linear dependency
        b[:, 1] = 2*b[:, 2]
    run_solver_test(pym.solvers.SolverDiagonal, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3, 4], ids=['singleRHS', 'columnRHS', 'multiRHS', 'multiRHS_lindep'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', [
    'mat_real_diagonal',
    'mat_real_spdiag',
    'mat_real_symm_pos_def',
    'mat_real_symm_pos_def_dynamic',
    'mat_real_symm_indef',
    'mat_real_symm_indef_dynamic',
    'mat_real_asymm',
    'mat_real_symm_saddle',
    'mat_complex_diagonal',
    'mat_complex_spdiag',
    'mat_complex_symm_pos_def_dynamic',
    'mat_complex_symm_indef_dynamic',
    'mat_complex_hermitian_1',
    'mat_complex_hermitian_indef',
    'mat_complex_hermitian_pos_def',
])
@pytest.mark.parametrize('mat_converter', mat_type_converters[:4], ids=mat_type_ids[:4])
def test_sparse_lu(Atag, b_type, b_shape, mat_converter):
    A = mat_converter(all_matrices[Atag])
    b = construct_b(b_type, b_shape, A.shape[0])
    if not np.iscomplexobj(A) and np.iscomplexobj(b):
        pytest.skip("complex b with real A")
    if b_shape == 4:  # Create a linear dependency
        b[:, 1] = 2*b[:, 2]
    run_solver_test(pym.solvers.SolverSparseLU, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3, 4], ids=['singleRHS', 'columnRHS', 'multiRHS', 'multiRHS_lindep'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', [
    'mat_real_diagonal',
    'mat_real_spdiag',
    'mat_real_symm_pos_def',
    'mat_real_symm_pos_def_dynamic',
    'mat_complex_hermitian_indef',
    'mat_complex_hermitian_1',
    'mat_complex_hermitian_pos_def',
])
@pytest.mark.parametrize('mat_converter', mat_type_converters[1:4], ids=mat_type_ids[1:4])
def test_sparse_cholesky_scikit(Atag, b_type, b_shape, mat_converter):
    # I don't know why, but scikit is able to solve indefinite matrix as well. Maybe they do some LDL inside?
    A = mat_converter(all_matrices[Atag])
    b = construct_b(b_type, b_shape, A.shape[0])
    if not np.iscomplexobj(A) and np.iscomplexobj(b):
        pytest.skip("complex b with real A")
    if b_shape == 4:  # Create a linear dependency
        b[:, 1] = 2*b[:, 2]
    run_solver_test(pym.solvers.SolverSparseCholeskyScikit, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3, 4], ids=['singleRHS', 'columnRHS', 'multiRHS', 'multiRHS_lindep'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', [
    'mat_real_diagonal',
    'mat_real_spdiag',
    'mat_real_symm_pos_def',
    'mat_real_symm_pos_def_dynamic',
    'mat_complex_hermitian_1',
    'mat_complex_hermitian_pos_def',
])
@pytest.mark.parametrize('mat_converter', mat_type_converters[:4], ids=mat_type_ids[:4])
def test_sparse_cholesky_cvxopt(Atag, b_type, b_shape, mat_converter):
    A = mat_converter(all_matrices[Atag])
    b = construct_b(b_type, b_shape, A.shape[0])
    if not np.iscomplexobj(A) and np.iscomplexobj(b):
        pytest.skip("complex b with real A")
    if b_shape == 4:  # Create a linear dependency
        b[:, 1] = 2*b[:, 2]
    run_solver_test(pym.solvers.SolverSparseCholeskyCVXOPT, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3, 4], ids=['singleRHS', 'columnRHS', 'multiRHS', 'multiRHS_lindep'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', [
        'mat_real_diagonal',
        'mat_real_spdiag',
        'mat_real_symm_pos_def',
        'mat_real_symm_pos_def_dynamic',
        'mat_real_symm_indef',
        'mat_real_symm_indef_dynamic',
        'mat_real_symm_saddle',
        'mat_real_asymm',
        # 'mat_complex_diagonal', # PyPardiso does not work for complex matrix yet
        # 'mat_complex_spdiag',
        # 'mat_complex_symm_pos_def_dynamic',
        # 'mat_complex_symm_indef_dynamic',
        # 'mat_complex_hermitian_1',
])
@pytest.mark.parametrize('mat_converter', mat_type_converters[:4], ids=mat_type_ids[:4])
def test_sparse_pardiso(Atag, b_type, b_shape, mat_converter):
    A = mat_converter(all_matrices[Atag])
    b = construct_b(b_type, b_shape, A.shape[0])
    if not np.iscomplexobj(A) and np.iscomplexobj(b):
        pytest.skip("complex b with real A")
    if b_shape == 4:  # Create a linear dependency
        b[:, 1] = 2*b[:, 2]
    run_solver_test(pym.solvers.SolverSparsePardiso, A, b)


@pytest.mark.parametrize('b_shape', [None, 1, 3, 4], ids=['singleRHS', 'columnRHS', 'multiRHS', 'multiRHS_lindep'])
@pytest.mark.parametrize('b_type', ['realRHS', 'complexRHS', 'imaginaryRHS'])
@pytest.mark.parametrize('Atag', [
        'mat_real_diagonal',
        'mat_real_spdiag',
        'mat_real_symm_pos_def',
        'mat_real_symm_pos_def_dynamic',
        # 'mat_real_symm_indef',
        # 'mat_real_symm_indef_dynamic',
        # 'mat_real_asymm',
        # 'mat_real_symm_saddle',
        'mat_complex_diagonal',
        'mat_complex_spdiag',
        # 'mat_complex_symm_pos_def_dynamic', # Works but needs many iterations
        # 'mat_complex_symm_indef_dynamic',
        'mat_complex_hermitian_1',
        # 'mat_complex_hermitian_indef',
        'mat_complex_hermitian_pos_def',
])
@pytest.mark.parametrize('mat_converter', mat_type_converters[:4], ids=mat_type_ids[:4])
def test_sparse_pardiso(Atag, b_type, b_shape, mat_converter):
    A = mat_converter(all_matrices[Atag])
    b = construct_b(b_type, b_shape, A.shape[0])
    if not np.iscomplexobj(A) and np.iscomplexobj(b):
        pytest.skip("complex b with real A")
    if b_shape == 4:  # Create a linear dependency
        b[:, 1] = 2*b[:, 2]
    # kwargs = dict(preconditioner=pym.solvers.ILU(), verbosity=1, tol=1e-10)
    kwargs = dict(preconditioner=pym.solvers.SOR(w=1.0), verbosity=1, tol=1e-10)
    # kwargs = dict(preconditioner=pym.solvers.DampedJacobi(), verbosity=1, tol=1e-10)
    run_solver_test(pym.solvers.CG, A, b, **kwargs)


if __name__ == '__main__':
    pytest.main()
