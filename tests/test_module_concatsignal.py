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


    def test_error_with_empty_state(self):
        s1 = pym.Signal('sig1', 0.0)
        s2 = pym.Signal('sig2', np.array([1.0, 2.0, 3.0, 4.0]))
        s3 = pym.Signal('sig3')
        s4 = pym.Signal('sig4', np.array([[6.0, 7.0]]))
        s5 = pym.Signal('sig5', np.array(8.0))

        # Get the state of the concatenated signal
        s = pym.SignalConcat(s1, s2, s3, s4, s5, tag='Concatenated')
        self.assertRaises(ValueError, lambda: s.state)

    def test_sensitivity(self):
        s1 = pym.Signal('sig1', 0.0)
        s2 = pym.Signal('sig2', np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 2.1, 3.1, 4.1]))
        s3 = pym.Signal('sig3', 5.0, 5.1)
        s4 = pym.Signal('sig4', np.array([[6.0, 7.0]]))
        s5 = pym.Signal('sig5', np.array(8.0), np.array(8.1))

        # Get the sensitivity of the concatenated signal
        s = pym.SignalConcat(s1, s2, s3, s4, s5, tag='Concatenated')
        self.assertTrue(np.allclose(s.sensitivity, np.array([0.0, 1.1, 2.1, 3.1, 4.1, 5.1, 0.0, 0.0, 8.1])))

        # Set the sensitivity of the concatenated signal
        s.sensitivity = np.array([8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1, 0.1])
        self.assertEqual(s1.sensitivity, 8.1)
        self.assertEqual(type(s1.sensitivity), float)

        self.assertTrue(np.allclose(s2.sensitivity, np.array([7.1, 6.1, 5.1, 4.1])))
        self.assertEqual(s2.sensitivity.shape, (4,))
        self.assertEqual(type(s2.sensitivity), np.ndarray)

        self.assertEqual(s3.sensitivity, 3.1)
        self.assertEqual(type(s3.sensitivity), float)

        self.assertTrue(np.allclose(s4.sensitivity, np.array([[2.1, 1.1]])))
        self.assertEqual(type(s4.sensitivity), np.ndarray)
        self.assertEqual(s4.sensitivity.shape, (1, 2))

        self.assertEqual(s5.sensitivity, np.array(0.1))
        self.assertEqual(s5.sensitivity.shape, ())
        self.assertEqual(type(s5.sensitivity), np.ndarray)

        # Try incrementing
        s.sensitivity += np.ones_like(s.state)
        self.assertTrue(np.allclose(s.sensitivity, np.array([9.1, 8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1])))

        s.add_sensitivity(np.ones_like(s.state))
        self.assertTrue(np.allclose(s.sensitivity, np.array([10.1, 9.1, 8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1])))

        # Try to slice the sensitivity and then set a new value (THIS DOESN'T WORK UNFORTUNATELY)
        # s.sensitivity[:4] = -1.0
        # self.assertTrue(np.allclose(s.sensitivity, np.array([-1.0, -1.0, -1.0, -1.0, 5.1, 4.1, 3.1, 2.1, 1.1])))

        # Reset (without allocation)
        s.reset(keep_alloc=False)
        self.assertEqual(s.sensitivity, None)
        self.assertEqual(s1.sensitivity, None)
        self.assertEqual(s2.sensitivity, None)
        self.assertEqual(s3.sensitivity, None)
        self.assertEqual(s4.sensitivity, None)
        self.assertEqual(s5.sensitivity, None)

        # Add sensitivity to empty
        s.add_sensitivity(np.ones_like(s.state))
        self.assertTrue(np.allclose(s.sensitivity, np.ones_like(s.state)))

        # Reset (With allocation)
        s.sensitivity = np.array([8.1, 7.1, 6.1, 5.1, 4.1, 3.1, 2.1, 1.1, 0.1])
        s.reset(keep_alloc=True)
        self.assertTrue(np.allclose(s.sensitivity, np.zeros((9,))))
        self.assertEqual(s1.sensitivity, 0.0)
        self.assertTrue(np.allclose(s2.sensitivity, np.zeros(4,)))
        self.assertEqual(s3.sensitivity, 0.0)
        self.assertTrue(np.allclose(s4.sensitivity, np.zeros((1,2))))
        self.assertTrue(np.allclose(s5.sensitivity, np.zeros(())))


if __name__ == '__main__':
    unittest.main()
