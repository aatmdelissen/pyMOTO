import unittest
import pymoto as pym
import numpy as np
import numpy.testing as npt


class TestActiveSet(unittest.TestCase):
    def test_inactive(self):
        """ No options given means all values are passed """
        np.random.seed(0)
        x = np.random.rand(5)
        actset = pym.AggActiveSet()
        self.assertEqual(x[actset(x)].size, 5)

    def test_all_equal(self):
        """ In case all values are equal they all must be passed """
        x = np.ones(5)
        actset = pym.AggActiveSet(0.1, 0.9, 0.1, 0.9)
        self.assertEqual(x[actset(x)].size, 5)

    def test_remove_below_value(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_rel=0.1)
        self.assertEqual(min(x[actset(x)]), 10)
        self.assertEqual(max(x[actset(x)]), 100)

    def test_remove_above_value(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(upper_rel=0.9)
        self.assertEqual(min(x[actset(x)]), 0)
        self.assertEqual(max(x[actset(x)]), 90)

    def test_remove_above_and_below_value(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_rel=0.1, upper_rel=0.9)
        self.assertEqual(min(x[actset(x)]), 10)
        self.assertEqual(max(x[actset(x)]), 90)

    def test_remove_lowest_amount(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_amt=0.15)
        self.assertEqual(min(x[actset(x)]), 15)
        self.assertEqual(max(x[actset(x)]), 100)

    def test_remove_highest_amount(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(upper_amt=0.95)
        self.assertEqual(min(x[actset(x)]), 0)
        self.assertEqual(max(x[actset(x)]), 95)

    def test_remove_lowest_and_highest_amount(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_amt=0.15, upper_amt=0.95)
        self.assertEqual(min(x[actset(x)]), 15)
        self.assertEqual(max(x[actset(x)]), 95)

    def test_remove_both_value_and_amount(self):
        x = np.arange(101)
        actset = pym.AggActiveSet(lower_rel=0.1, upper_rel=0.9, lower_amt=0.15, upper_amt=0.95)
        self.assertEqual(min(x[actset(x)]), 15)
        self.assertEqual(max(x[actset(x)]), 90)


class TestScaling(unittest.TestCase):
    def test_min_no_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('min')
        sf = sc(x, 1.0)
        self.assertEqual(min(x), sf)
        x = np.random.rand(5)
        sf = sc(x, 2.0)
        self.assertEqual(min(x), 2.0*sf)

    def test_max_no_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('max')
        sf = sc(x, 1.0)
        self.assertEqual(max(x), sf)
        x = np.random.rand(5)
        sf = sc(x, 2.0)
        self.assertEqual(max(x), 2.0 * sf)

    def test_min_full_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('min', damping=1.0)
        sf0 = sc(x, 1.0)
        self.assertEqual(min(x), sf0)
        x = np.random.rand(5)
        sf = sc(x, 2.0)
        self.assertEqual(sf, sf0)

    def test_max_full_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('max', damping=1.0)
        sf0 = sc(x, 1.0)
        self.assertEqual(max(x), sf0)
        x = np.random.rand(5)
        sf = sc(x, 2.0)
        self.assertEqual(sf, sf0)

    def test_min_half_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('min', damping=0.5)
        sf0 = sc(x, 1.0)
        self.assertEqual(min(x), sf0)
        x = np.random.rand(5)

        for i in range(10):  # Converges toward minimum
            sf = sc(x, 1.0)
            self.assertLess(sf - min(x), sf0 - min(x))
            sf0 = sf

    def test_max_half_damping(self):
        np.random.seed(0)
        x = np.random.rand(5)
        sc = pym.AggScaling('max', damping=0.5)
        sf0 = sc(x, 1.0)
        self.assertEqual(max(x), sf0)
        x = np.random.rand(5)

        for i in range(10):  # Converges toward minimum
            sf = sc(x, 1.0)
            self.assertLess(max(x) - sf, max(x) - sf0)
            sf0 = sf


class TestPNorm(unittest.TestCase):
    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

    def test_PNorm_max_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        m = pym.PNorm(sx, p=2)

        pym.finite_difference(m, test_fn=self.fd_testfn)

    def test_PNorm_max1_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        m = pym.PNorm(sx, p=20)

        pym.finite_difference(m, test_fn=self.fd_testfn)

    def test_PNorm_min_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        m = pym.PNorm(sx, p=-4)

        pym.finite_difference(m, test_fn=self.fd_testfn)


class TestSoftMinMax(unittest.TestCase):
    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

    def test_soft_max_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        m = pym.SoftMinMax(sx, alpha=2)

        pym.finite_difference(m, test_fn=self.fd_testfn)

    def test_soft_min_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        m = pym.SoftMinMax(sx, alpha=-20)

        pym.finite_difference(m, test_fn=self.fd_testfn)


class TestKS(unittest.TestCase):
    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

    def test_KS_max_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        m = pym.KSFunction(sx, rho=2)
        m.response()
        self.assertGreaterEqual(m.sig_out[0].state, max(sx.state))

        pym.finite_difference(m, test_fn=self.fd_testfn)

        # Converge to actual maximum
        y_2 = m.sig_out[0].state
        m.rho = 3
        m.response()
        y_3 = m.sig_out[0].state
        self.assertLessEqual(y_3 - max(sx.state), y_2 - max(sx.state))

    def test_KS_min_fd(self):
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(100))
        m = pym.KSFunction(sx, rho=-20)

        pym.finite_difference(m, test_fn=self.fd_testfn)