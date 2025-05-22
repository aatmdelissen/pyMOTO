import pytest
import numpy.testing as npt
import pymoto as pym
import numpy as np


class TestNetwork:
    def test_network_wsignals(self):
        x1 = pym.Signal('x1', 2.0)
        x2 = pym.Signal('x2', 3.0)

        with pym.Network() as netw:
            y1 = pym.MathGeneral("x1*2.0")(x1)
            y1.tag = 'y1'
            y2 = pym.MathGeneral("x2*x2 + 2.0")(x2)
            y2.tag = 'y2'
            z = pym.MathGeneral("y1*y2")(y1, y2)

        netw.response()
        assert y1.state == 4.0
        assert y2.state == 11.0
        assert z.state == 44.0

        z.sensitivity = 1.0
        netw.sensitivity()
        assert y1.sensitivity == 11.0
        assert y2.sensitivity == 4.0
        assert x1.sensitivity == 22.0
        assert x2.sensitivity == 24.0

        netw.reset()
        assert x1.sensitivity is None
        assert x2.sensitivity is None
        assert y1.sensitivity is None
        assert y2.sensitivity is None
        assert z.sensitivity is None

    def test_reconnect_module_add_twice_in_network(self):
        class MyMod(pym.Module):
            def __call__(self, A, B):
                return A + B

            def _sensitivity(self, dC):
                return dC, dC

        s_A = pym.Signal('A', np.array([1, 2, 3]))
        s_B = pym.Signal('B', np.array([4, 5, 6]))
        s_C = pym.Signal('C', np.array([1, 2]))
        s_D = pym.Signal('D', np.array([4, 5]))
        m = MyMod()
        with pym.Network() as fn:
            s_E = m(s_A, s_B)
            pytest.raises(RuntimeError, m, s_C, s_D)

    def test_reconnect_module_add_twice_in_network_copy(self):
        class MyMod(pym.Module):
            def __call__(self, A, B):
                return A + B

            def _sensitivity(self, dC):
                return dC, dC

        s_A = pym.Signal('A', np.array([1, 2, 3]))
        s_B = pym.Signal('B', np.array([4, 5, 6]))
        s_C = pym.Signal('C', np.array([1, 2]))
        s_D = pym.Signal('D', np.array([4, 5]))
        m = MyMod()
        import copy
        with pym.Network() as fn:
            s_E = m(s_A, s_B)
            s_F = copy.deepcopy(m)(s_C, s_D)  # Use copy to create a new module
        npt.assert_equal(s_E.state, np.array([5, 7, 9]))
        npt.assert_equal(s_F.state, np.array([5, 7]))

