import pytest
import numpy as np
import pymoto as pym
import numpy.testing as npt


class TestScaling:
    np.random.seed(0)

    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-6, atol=1e-15)

    def test_objective(self):
        sx = pym.Signal('x', 1.0)
        m = pym.Scaling(scaling=105.0)
        s_scaled = m(sx)

        assert s_scaled.state == 1.0 * 105.0
        sx.state = 2.0
        m.response()
        assert s_scaled.state == 2.0 * 105.0

        pym.finite_difference(sx, s_scaled, test_fn=self.fd_testfn)

    def test_negative_objective(self):
        """ Test if the negative sign is kept"""
        sx = pym.Signal('x', -1.0)
        m = pym.Scaling(scaling=105.0)
        s_scaled = m(sx)

        assert s_scaled.state == -1.0 * 105.0

        pym.finite_difference(sx, s_scaled, test_fn=self.fd_testfn)

    @pytest.mark.parametrize("minval", [-0.5, 0.0, 0.5])
    def test_lower_constraint(self, minval):
        sx = pym.Signal('x', 0.0)
        m = pym.Scaling(scaling=15.0, minval=minval)
        s_scaled = m(sx)

        x_values = minval + np.linspace(-1, 1, 21)
        for x in x_values:
            sx.state = x
            m.response()
            if x < minval:
                assert s_scaled.state > 0.0
            elif x == minval:
                assert s_scaled.state == 0.0
            else:
                assert s_scaled.state < 0.0

        sx.state = minval - (abs(minval) if minval != 0 else 1.0)
        m.response()
        assert s_scaled.state == 15.0

        pym.finite_difference(sx, s_scaled, test_fn=self.fd_testfn)

    @pytest.mark.parametrize("maxval", [-0.5, 0.0, 0.5])
    def test_upper_constraint(self, maxval):
        sx = pym.Signal('x', 0.0)
        m = pym.Scaling(scaling=15.0, maxval=maxval)
        s_scaled = m(sx)

        x_values = maxval + np.linspace(-1, 1, 21)
        for x in x_values:
            sx.state = x
            m.response()
            if x < maxval:
                assert s_scaled.state < 0.0
            elif x == maxval:
                assert s_scaled.state == 0.0
            else:
                assert s_scaled.state > 0.0

        sx.state = maxval + (abs(maxval) if maxval != 0 else 1.0)
        m.response()
        assert s_scaled.state == 15.0

        pym.finite_difference(sx, s_scaled, test_fn=self.fd_testfn)

    @pytest.mark.parametrize("minval,maxval", [(-0.5, -0.1), (-0.5, 0.0), (-0.5, 0.5), (0.0, 0.5), (0.1, 0.5)])
    def test_double_constraint(self, minval, maxval):
        sx = pym.Signal('x', 0.0)
        m = pym.Scaling(scaling=15.0, minval=minval, maxval=maxval)
        s_scaled = m(sx)

        x_values = maxval + np.linspace(-1, 1, 21)
        for x in x_values:
            sx.state = x
            m.response()
            if minval < x < maxval:
                assert s_scaled.state < 0.0
            elif x == maxval or x == minval:
                npt.assert_allclose(s_scaled.state, 0.0, atol=1e-14)
            else:
                assert s_scaled.state > 0.0

        pym.finite_difference(sx, s_scaled, test_fn=self.fd_testfn)
    

if __name__ == '__main__':
    pytest.main([__file__])
