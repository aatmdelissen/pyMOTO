import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym
    

class TestAutoMod:
    def test_automod_scalar(self):
        def resp_fn(x, y):
            return np.sum(x*y)
            
        try:
            m = pym.AutoMod(resp_fn)
        except ImportError:
            pytest.skip("No JAX available")
            
        sx = pym.Signal("x", 1.1)
        sy = pym.Signal("y", 2.0)

        sz = m(sx, sy)
        sz.tag = "z"

        assert sz.state == 1.1*2.0

        sz.sensitivity = 1.5
        m.sensitivity()

        npt.assert_allclose(sx.sensitivity, 1.5*2.0)
        npt.assert_allclose(sy.sensitivity, 1.5*1.1)

    def test_automod_vec(self):
        def resp_fn(x, y):
            return np.sum(x*y)
            
        try:
            m = pym.AutoMod(resp_fn)
        except ImportError:
            pytest.skip("No JAX available")
            
        sx = pym.Signal("x", np.array([1.1, 1.2, 1.3]))
        sy = pym.Signal("y", np.array([2.5, 2.6, 2.7]))

        sz = m(sx, sy)
        sz.tag = "z"

        assert sz.state == 1.1*2.5 + 1.2*2.6 + 1.3*2.7

        sz.sensitivity = 1.5
        m.sensitivity()

        npt.assert_allclose(sx.sensitivity, 1.5*sy.state)
        npt.assert_allclose(sy.sensitivity, 1.5*sx.state)


if __name__ == '__main__':
    pytest.main()
