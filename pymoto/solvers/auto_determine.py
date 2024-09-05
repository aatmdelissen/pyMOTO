import warnings
import scipy.sparse as sps
from inspect import currentframe, getframeinfo

from .dense import *
from .sparse import *
from .matrix_checks import *


# flake8: noqa: C901
def auto_determine_solver(A, isdiagonal=None, islowertriangular=None, isuppertriangular=None,
                          ishermitian=None, issymmetric=None, ispositivedefinite=None):
    """
    Uses parts of Matlab's scheme https://nl.mathworks.com/help/matlab/ref/mldivide.html
    :param A: The matrix
    :param isdiagonal: Manual override for diagonal matrix
    :param islowertriangular: Override for lower triangular matrix
    :param isuppertriangular: Override for upper triangular matrix
    :param ishermitian: Override for hermitian matrix (prevents check)
    :param issymmetric: Override for symmetric matrix (prevents check). Is the same as hermitian for a real matrix
    :param ispositivedefinite: Manual override for positive definiteness
    :return: LinearSolver which should be 'best' for the matrix
    """
    issparse = sps.issparse(A)  # Check if the matrix is sparse
    issquare = A.shape[0] == A.shape[1]  # Check if the matrix is square

    if not issquare:
        if issparse:
            sps.SparseEfficiencyWarning("Only a dense version of QR solver is available")  # TODO
        return SolverDenseQR()

    # l_bw, u_bw = spla.bandwidth(A) # TODO Get bandwidth (implemented in scipy version > 1.8.0)

    if isdiagonal is None:  # Check if matrix is diagonal
        # TODO: This could be improved to check other sparse matrix types as well
        isdiagonal = matrix_is_diagonal(A)
    if isdiagonal:
        return SolverDiagonal()

    # Check if the matrix is triangular
    # TODO Currently only for dense matrices
    if islowertriangular is None:  # Check if matrix is lower triangular
        islowertriangular = False if issparse else np.allclose(A, np.tril(A))
    if islowertriangular:
        warnings.WarningMessage("Lower triangular solver not implemented",
                                UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

    if isuppertriangular is None:  # Check if matrix is upper triangular
        isuppertriangular = False if issparse else np.allclose(A, np.triu(A))
    if isuppertriangular:
        warnings.WarningMessage("Upper triangular solver not implemented",
                                UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

    ispermutedtriangular = False
    if ispermutedtriangular:
        warnings.WarningMessage("Permuted triangular solver not implemented",
                                UserWarning, getframeinfo(currentframe()).filename, getframeinfo(currentframe()).lineno)

    # Check if the matrix is complex-valued
    iscomplex = np.iscomplexobj(A)
    if iscomplex:
        # Detect if the matrix is hermitian and/or symmetric
        if ishermitian is None:
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
        # Prefer Intel Pardiso solver as it can solve any matrix TODO: Check for complex matrix
        if SolverSparsePardiso.defined and not iscomplex:
            # TODO check for positive definiteness?  np.all(A.diagonal() > 0) or np.all(A.diagonal() < 0)
            return SolverSparsePardiso(symmetric=issymmetric, hermitian=ishermitian, positive_definite=ispositivedefinite)

        if ishermitian:
            # Check if diagonal is all positive or all negative -> Cholesky
            if ispositivedefinite is None:
                ispositivedefinite = np.all(A.diagonal() > 0) or np.all(A.diagonal() < 0)
            if ispositivedefinite:  # TODO what about the complex case?
                if SolverSparseCholeskyScikit.defined:
                    return SolverSparseCholeskyScikit()
                if SolverSparseCholeskyCVXOPT.defined:
                    return SolverSparseCholeskyCVXOPT()

        return SolverSparseLU()  # Default to LU, which should be possible for any non-singular square matrix

    else:  # Dense branch
        if ishermitian:
            # Check if diagonal is all positive or all negative
            if np.all(A.diagonal() > 0) or np.all(A.diagonal() < 0):
                return SolverDenseCholesky()
            else:
                return SolverDenseLDL(hermitian=ishermitian)
        elif issymmetric:
            return SolverDenseLDL(hermitian=ishermitian)
        else:
            # TODO: Detect if the matrix is Hessenberg
            return SolverDenseLU()
