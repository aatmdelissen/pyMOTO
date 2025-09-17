import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym
    
def fd_testfn(x0, dx, df_an, df_fd):
    npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

def id_fn(val):
    npval = np.asarray(val)
    has_r = np.abs(npval.real).max() > 0
    has_i = np.abs(npval.imag).max() > 0
    if has_i and not has_r:
        valuetype = "imaginary"
    elif not has_i and has_r:
        valuetype = "real"
    else:
        valuetype = "complex"

    dimensionality = "vector" if np.val.size > 1 else "scalar"
    return f"{valuetype} {dimensionality}"

@pytest.mark.parametrize('x0', [1.1, 1.1 + 2.0j, 2.0j, 
                                np.array([1.1, 1.2, 1.3]), 
                                np.array([1.1 + 1j, 1.2 + 2j, 1.3 + 3j]), 
                                np.array([1.1j, 1.2j, 1.3j])], ids=id_fn)
@pytest.mark.parametrize('y0', [2.0, 2.0 + 1.3j, 1.3j, 
                                np.array([2.5, 2.6, 2.7]), 
                                np.array([2.5 + 2j, 2.6 + 1j, 2.7 + 4j]), 
                                np.array([2.5j, 2.6j, 2.7j])], ids=id_fn)
@pytest.mark.parametrize('backend', ['autograd', 'jax'])
def test_automod_one_output(backend, x0, y0):
    def resp_fn(x, y):
        return x * y

    try:
        m = pym.AutoMod(resp_fn, backend=backend)
    except ImportError:
        pytest.skip(f"Backend `{backend}` not available")
        
    sx = pym.Signal("x", x0)
    sy = pym.Signal("y", y0)

    sz = m(sx, sy)
    sz.tag = "z"

    npt.assert_allclose(sz.state, resp_fn(sx.state, sy.state))

    pym.finite_difference([sx, sy], sz, test_fn=fd_testfn)


@pytest.mark.parametrize('x0', [1.1, 1.1 + 2.0j, 2.0j], ids=id_fn)
@pytest.mark.parametrize('y0', [np.array([2.5, 2.6, 2.7]), 
                                np.array([2.5 + 2j, 2.6 + 1j, 2.7 + 4j]), 
                                np.array([2.5j, 2.6j, 2.7j])], 
                         ids=id_fn)
@pytest.mark.parametrize('backend', ['autograd', 'jax'])
def test_automod_vec_2out(backend, x0, y0):
    def resp_fn(x, y):
        return x + y, x*y

    sx = pym.Signal("x", x0)
    sy = pym.Signal("y", y0)

    try:
        m = pym.AutoMod(resp_fn, backend=backend)
        sz1, sz2 = m(sx, sy)
    except ImportError:
        pytest.skip(f"Backend `{backend}` not available")
    except NotImplementedError:
        pytest.skip("Two outputs not supported")


    sz1.tag = "z1"
    sz2.tag = "z2"

    out1, out2 = resp_fn(sx.state, sy.state)
    npt.assert_allclose(sz1.state, out1)
    npt.assert_allclose(sz2.state, out2)

    pym.finite_difference([sx, sy], [sz1, sz2], test_fn=fd_testfn)


if __name__ == '__main__':
    pytest.main()
