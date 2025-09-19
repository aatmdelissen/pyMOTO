import pytest
import pymoto as pym
import numpy as np
import numpy.testing as npt


class TestActiveSet:
    def test_inactive(self):
        """ No options given means all values are passed """
        np.random.seed(0)
        x = np.random.rand(5)
        actset = pym.AggActiveSet()
        assert x[actset(x)].size == 5

    def test_all_equal(self):
        """ In case all values are equal they all must be passed """
        x = np.ones(5)
        actset = pym.AggActiveSet(0.1, 0.9, 0.1, 0.9)
        assert x[actset(x)].size == 5

    def test_remove_below_value(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_rel=0.1)
        assert min(x[actset(x)]) == 10
        assert max(x[actset(x)]) == 100

    def test_remove_above_value(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(upper_rel=0.9)
        assert min(x[actset(x)]) == 0
        assert max(x[actset(x)]) == 90

    def test_remove_above_and_below_value(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_rel=0.1, upper_rel=0.9)
        assert min(x[actset(x)]) == 10
        assert max(x[actset(x)]) == 90

    def test_remove_lowest_amount(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_amt=0.15)
        assert min(x[actset(x)]) == 15
        assert max(x[actset(x)]) == 100

    def test_remove_highest_amount(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(upper_amt=0.95)
        assert min(x[actset(x)]) == 0
        assert max(x[actset(x)]) == 95

    def test_remove_lowest_and_highest_amount(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_amt=0.15, upper_amt=0.95)
        assert min(x[actset(x)]) == 15
        assert max(x[actset(x)]) == 95

    def test_remove_both_value_and_amount(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_rel=0.1, upper_rel=0.9, lower_amt=0.15, upper_amt=0.95)
        assert min(x[actset(x)]) == 15
        assert max(x[actset(x)]) == 90


class TestScaling:
    def test_min_no_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('min')
        sf = sc(x, 1.0)
        assert min(x) == sf
        x = np.random.rand(5)
        sf = sc(x, 2.0)
        assert min(x) == 2.0*sf

    def test_max_no_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('max')
        sf = sc(x, 1.0)
        assert max(x) == sf
        x = np.random.rand(5)
        sf = sc(x, 2.0)
        assert max(x) == 2.0*sf

    def test_min_full_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('min', damping=1.0)
        sf0 = sc(x, 1.0)
        assert min(x) == sf0
        x = np.random.rand(5)
        sf = sc(x, 2.0)
        assert sf == sf0

    def test_max_full_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('max', damping=1.0)
        sf0 = sc(x, 1.0)
        assert max(x) == sf0
        x = np.random.rand(5)
        sf = sc(x, 2.0)
        assert sf == sf0

    def test_min_half_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('min', damping=0.5)
        sf0 = sc(x, 1.0)
        assert min(x) == sf0
        x = np.random.rand(5)

        for i in range(10):  # Converges toward minimum
            sf = sc(x, 1.0)
            assert sf - min(x) < sf0 - min(x)
            sf0 = sf

    def test_max_half_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('max', damping=0.5)
        sf0 = sc(x, 1.0)
        assert max(x) == sf0
        x = np.random.rand(5)

        for i in range(10):  # Converges toward minimum
            sf = sc(x, 1.0)
            assert max(x) - sf < max(x) - sf0
            sf0 = sf


class TestPNorm:
    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

    def test_PNorm_max_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        sy = pym.PNorm(p=2)(sx)

        pym.finite_difference(sx, sy, test_fn=self.fd_testfn)

    def test_PNorm_max1_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        sy = pym.PNorm(p=20)(sx)

        pym.finite_difference(sx, sy, test_fn=self.fd_testfn)

    def test_PNorm_min_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        sy = pym.PNorm(p=-4)(sx)

        pym.finite_difference(sx, sy, test_fn=self.fd_testfn)


class TestSoftMinMax:
    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

    def test_soft_max_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        sy = pym.SoftMinMax(alpha=2)(sx)

        pym.finite_difference(sx, sy, test_fn=self.fd_testfn)

    def test_soft_min_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        sy = pym.SoftMinMax(alpha=-20)(sx)

        pym.finite_difference(sx, sy, test_fn=self.fd_testfn)


class TestKS:
    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

    def test_KS_max_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        m = pym.KSFunction(rho=2)
        sy = m(sx)

        assert sy.state >= max(sx.state)

        pym.finite_difference(sx, sy, test_fn=self.fd_testfn)

        # Converge to actual maximum
        y_2 = sy.state
        m.rho = 3
        m.response()
        y_3 = sy.state
        assert y_3 - max(sx.state) <= y_2 - max(sx.state)

    def test_KS_min_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        sy = pym.KSFunction(rho=-20)(sx)

        pym.finite_difference(sx, sy, test_fn=self.fd_testfn)


if __name__ == '__main__':
    pytest.main([__file__])
