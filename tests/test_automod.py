import unittest
import numpy as np
import pymoto as pym


class TestAutoMod(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """ Skip test if JAX is not installed """
        try:
            import jax
        except ImportError:
            raise unittest.SkipTest(f"Skipping test {cls}")

    def test_automod_scalar(self):
        class Mult(pym.AutoMod):
            def _response(self, x, y):
                return x * y

        sx = pym.Signal("x", 1.1)
        sy = pym.Signal("y", 2.0)

        m = Mult([sx, sy], pym.Signal("z"))

        m.response()

        self.assertAlmostEqual(m.sig_out[0].state, 1.1*2.0)

        m.sig_out[0].sensitivity = 1.5

        m.sensitivity()

        self.assertAlmostEqual(sx.sensitivity, 1.5*2.0, 5)
        self.assertAlmostEqual(sy.sensitivity, 1.5*1.1, 5)

    def test_automod_vec(self):
        class MultSum(pym.AutoMod):
            def _response(self, x, y):
                return np.sum(x*y)

        sx = pym.Signal("x", np.array([1.1, 1.2, 1.3]))
        sy = pym.Signal("y", np.array([2.5, 2.6, 2.7]))

        m = MultSum([sx, sy], pym.Signal("z"))

        m.response()

        self.assertAlmostEqual(m.sig_out[0].state, 1.1*2.5 + 1.2*2.6 + 1.3*2.7)

        m.sig_out[0].sensitivity = 1.5

        m.sensitivity()

        self.assertTrue(np.allclose(sx.sensitivity, 1.5*sy.state))
        self.assertTrue(np.allclose(sy.sensitivity, 1.5*sx.state))


if __name__ == '__main__':
    unittest.main()
