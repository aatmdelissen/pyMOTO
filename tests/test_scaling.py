import unittest
import numpy as np
import pymoto as pym


class TestScaling(unittest.TestCase):
    def test_objective(self):
        sx = pym.Signal('x', 1.0)
        m = pym.Scaling(sx, scaling=105.0)
        m.response()
        self.assertEqual(m.sig_out[0].state, 105.0)
        sx.state = 2.0
        m.response()
        self.assertEqual(m.sig_out[0].state, 210.0)

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-7, atol=1e-5))
        pym.finite_difference(m, test_fn=tfn)

    def test_lower_constraint(self):
        sx = pym.Signal('x', 1.0)
        m = pym.Scaling(sx, scaling=15.0, minval=0.5)
        m.response()
        self.assertLess(m.sig_out[0].state, 0.0)
        sx.state = 0.5
        m.response()
        self.assertEqual(m.sig_out[0].state, 0.0)
        sx.state = 0.4
        m.response()
        self.assertGreater(m.sig_out[0].state, 0.0)
        sx.state = 0.0
        m.response()
        self.assertEqual(m.sig_out[0].state, 15.0)

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-7, atol=1e-5))
        pym.finite_difference(m, test_fn=tfn)

    def test_upper_constraint(self):
        sx = pym.Signal('x', 1.0)
        m = pym.Scaling(sx, scaling=15.0, maxval=0.5)
        m.response()
        self.assertGreater(m.sig_out[0].state, 0.0)
        sx.state = 0.5
        m.response()
        self.assertEqual(m.sig_out[0].state, 0.0)
        sx.state = 0.4
        m.response()
        self.assertLess(m.sig_out[0].state, 0.0)
        sx.state = 1.0
        m.response()
        self.assertEqual(m.sig_out[0].state, 15.0)

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-7, atol=1e-5))
        pym.finite_difference(m, test_fn=tfn)