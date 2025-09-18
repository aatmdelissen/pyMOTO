import pytest
import scipy.sparse as sps
import numpy as np
import numpy.testing as npt
import pymoto as pym


class MatrixGenerator(pym.Module):
    def __init__(self, mat):
        self.mat = mat

    def __call__(self, values):
        self.mat.data = values.copy()
        return self.mat
    
    def _sensitivity(self, dA: pym.DyadCarrier):
        dA_array = dA.toarray()
        return dA_array[self.mat.row, self.mat.col]


class MatrixSum(pym.Module):
    def __call__(self, A):
        return A.sum()

    def _sensitivity(self, dAsum):
        A = self.get_input_states()
        return pym.DyadCarrier(np.ones(A.shape[0]), np.ones(A.shape[1]), shape=A.shape) * dAsum
    

def fd_testfn(x0, dx, df_an, df_fd):
    npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)


@pytest.mark.parametrize('n_mat', [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize('complex_matrix', [False, True])
@pytest.mark.parametrize('complex_value', [False, True])
def test_add_matrices(n_mat: int, complex_matrix: bool, complex_value: bool):
    m, n = 10, 20
    sigs_mat = []
    sigs_fd_in = []
    Y_chk = 0
    for i in range(n_mat):
        a = np.random.rand(1).item()
        if i%2==0 and complex_value:
            a = a + 1j*np.random.rand(1).item()
        A = sps.random(m, n)
        if i%3==0 and complex_matrix:
            A.data = A.data + 1j*np.random.rand(A.nnz)
        Y_chk = Y_chk + a*A
        s_a = pym.Signal(f'a{i}', a)
        s_Avals = pym.Signal(f'Adat{i}', A.data.copy())
        s_A = MatrixGenerator(A)(s_Avals)
       
        sigs_mat.append(s_a)
        sigs_mat.append(s_A)
        sigs_fd_in.append(s_a)
        sigs_fd_in.append(s_Avals)
    
    sY = pym.AddMatrix()(*sigs_mat)
    assert (sY.state - Y_chk).nnz == 0

    sYsum = MatrixSum()(sY)

    pym.finite_difference(fromsig=sigs_fd_in, tosig=sYsum, test_fn=fd_testfn)


if __name__ == '__main__':
    pytest.main([__file__])
