import unittest
import numpy as np
import pymoto as pym


class TestConcat(unittest.TestCase):
    def test_concatsignal(self):
        s1 = pym.Signal('sig1', 0.0)
        s2 = pym.Signal('sig2', np.array([1.0, 2.0, 3.0, 4.0]))
        s3 = pym.Signal('sig3', 5.0)
        s4 = pym.Signal('sig4', np.array([[6.0, 7.0]]))
        s5 = pym.Signal('sig5', np.array(8.0))

        s = pym.Signal('out')
        m = pym.ConcatSignal([s1, s2, s3, s4, s5], s)
        m.response()

        self.assertTrue(np.allclose(s.state, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])))

        # Set the state of the concatenated signal
        s.sensitivity = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        m.sensitivity()
        self.assertEqual(s1.sensitivity, 8.0)
        self.assertEqual(type(s1.sensitivity), float)

        self.assertTrue(np.allclose(s2.sensitivity, np.array([7.0, 6.0, 5.0, 4.0])))
        self.assertEqual(s2.sensitivity.shape, (4,))
        self.assertEqual(type(s2.sensitivity), np.ndarray)

        self.assertEqual(s3.sensitivity, 3.0)
        self.assertEqual(type(s3.sensitivity), float)

        self.assertTrue(np.allclose(s4.sensitivity, np.array([[2.0, 1.0]])))
        self.assertEqual(type(s4.sensitivity), np.ndarray)
        self.assertEqual(s4.sensitivity.shape, (1, 2))

        self.assertEqual(s5.sensitivity, np.array(0.0))
        self.assertEqual(s5.sensitivity.shape, ())
        self.assertEqual(type(s5.sensitivity), np.ndarray)


if __name__ == '__main__':
    unittest.main()
