import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym


def fd_testfn(rtol=1e-5, atol=1e-15):
    def tfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=rtol, atol=atol)
    return tfn


@pytest.mark.parametrize('index, value', [(2, 3.14), 
                                          ([2, 3], 3.14), 
                                          ([2, 3], [3.8, 4.2]),
                                          (slice(1, 3), 1.0),
                                          (slice(1, 3), np.array([8.0, 3.4]))])
def test_set_vector(index, value):
    x = pym.Signal('x', np.array([1.0, 2.0, 3.0, 4.0]))
    y = pym.SetValue(index, value)(x)
    npt.assert_allclose(y.state[index], value)
    pym.finite_difference(x, y, test_fn=fd_testfn())


@pytest.mark.parametrize('index, value', [((2, 2), 3.14), 
                                          (([2, 2], [2, 3]), 3.14), 
                                          (([2, 2], [2, 3]), [3.8, 4.2]),
                                          ((slice(1, 3), [2, 3]), 1.0),
                                          ((slice(1, 3), slice(1, 4)), np.array([[8.0, 3.4, 1.3], [4.6, .2, 3.6]]))])
def test_set_matrix(index, value):
    x = pym.Signal('x', np.random.rand(3, 4))
    y = pym.SetValue(index, value)(x)
    npt.assert_allclose(y.state[index], value)
    pym.finite_difference(x, y, test_fn=fd_testfn())


if __name__ == '__main__':
    pytest.main([__file__])
