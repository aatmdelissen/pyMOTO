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

    def test_lower_constraint(self):
        sx = pym.Signal('x', 1.0)
        m = pym.Scaling(scaling=15.0, minval=0.5)
        s_scaled = m(sx)
        assert s_scaled.state < 0.0
        sx.state = 0.5
        m.response()
        assert s_scaled.state == 0.0
        sx.state = 0.4
        m.response()
        assert s_scaled.state > 0.0
        sx.state = 0.0
        m.response()
        assert s_scaled.state == 15.0

        pym.finite_difference(sx, s_scaled, test_fn=self.fd_testfn)

    def test_upper_constraint(self):
        sx = pym.Signal('x', 1.0)
        m = pym.Scaling(scaling=15.0, maxval=0.5)
        s_scaled = m(sx)
        assert s_scaled.state > 0.0
        sx.state = 0.5
        m.response()
        assert s_scaled.state == 0.0
        sx.state = 0.4
        m.response()
        assert s_scaled.state < 0.0
        sx.state = 1.0
        m.response()
        assert s_scaled.state == 15.0

        pym.finite_difference(sx, s_scaled, test_fn=self.fd_testfn)


if __name__ == '__main__':
    pytest.main([__file__])
