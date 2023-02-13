import unittest
import numpy as np
import pymoto as pym


class TestMath(unittest.TestCase):
    def testVec_Scalar(self):
        """ Test functionality where a vector is multiplied with a scalar """
        sVec = pym.Signal("vec", np.array([1.0, 0.0, 3.8, 4.6]))
        sScalar = pym.Signal("scalar", 3.5)

        sRes = pym.Signal("result")

        mod = pym.MathGeneral([sVec, sScalar], sRes, expression="inp0*inp1")

        mod.response()

        # Check result
        self.assertIsInstance(sRes.state, type(sVec.state))
        self.assertTrue(np.allclose(sRes.state, np.array([1.0*3.5, 0.0*3.5, 3.8*3.5, 4.6*3.5])))

        sRes.sensitivity = np.array([1.0, -2.0, 0.0, -4.0])

        mod.sensitivity()

        # Check sensitivity types
        self.assertIsInstance(sScalar.sensitivity, type(sScalar.state))
        self.assertIsInstance(sVec.sensitivity, type(sVec.state))

        # Check results
        self.assertAlmostEqual(sScalar.sensitivity, 1.0*1.0 - 0.0*2.0 + 3.8*0.0 - 4.6*4.0)
        self.assertTrue(np.allclose(sVec.sensitivity, np.array([1.0*3.5, -2.0*3.5, 0.0*3.5, -4.0*3.5])))

    def testVec_npScalar(self):
        """ Test functionality where a vector is multiplied with a scalar """
        sVec = pym.Signal("vec", np.array([1.0, 0.0, 3.8, 4.6]))
        sScalar = pym.Signal("numpy_scalar", np.array(3.5))

        sRes = pym.Signal("result")

        mod = pym.MathGeneral([sVec, sScalar], sRes, expression="inp0*inp1")

        mod.response()

        # Check result
        self.assertIsInstance(sRes.state, type(sVec.state))
        self.assertTrue(np.allclose(sRes.state, np.array([1.0*3.5, 0.0*3.5, 3.8*3.5, 4.6*3.5])))

        sRes.sensitivity = np.array([1.0, -2.0, 0.0, -4.0])

        mod.sensitivity()

        # Check sensitivity types
        self.assertIsInstance(sScalar.sensitivity, type(sScalar.state))
        self.assertIsInstance(sVec.sensitivity, type(sVec.state))

        # Check results
        self.assertAlmostEqual(sScalar.sensitivity, 1.0*1.0 - 0.0*2.0 + 3.8*0.0 - 4.6*4.0)
        self.assertTrue(np.allclose(sVec.sensitivity, np.array([1.0*3.5, -2.0*3.5, 0.0*3.5, -4.0*3.5])))

    def testVec_scalar_complex(self):
        """ Test functionality where a vector is multiplied with a complex scalar """
        v = np.array([1.0, 0.0+2.1j, 3.8, 4.6+3.6j])
        s = 3.5 + 3.8j
        sVec = pym.Signal("vec", v.copy())
        sScalar = pym.Signal("scalar", s)

        sRes = pym.Signal("result")

        mod = pym.MathGeneral([sVec, sScalar], sRes, expression="inp0*inp1")

        mod.response()

        # Check result
        self.assertIsInstance(sRes.state, type(sVec.state))
        self.assertTrue(np.allclose(sRes.state, v*s))

        dr = np.array([1.0+0.1j, -2.0-2.5j, 0.0+0.8j, -4.0-2.2j])
        sRes.sensitivity = dr.copy()

        mod.sensitivity()

        # Check sensitivity types
        self.assertIsInstance(sScalar.sensitivity, type(sScalar.state))
        self.assertIsInstance(sVec.sensitivity, type(sVec.state))
        self.assertEqual(sVec.sensitivity.dtype, sVec.state.dtype)

        # Check results
        self.assertAlmostEqual(sScalar.sensitivity, dr@v)
        self.assertTrue(np.allclose(sVec.sensitivity, dr*s))

    def testVec_scalar_complex_real_input(self):
        """ Test functionality where a vector is multiplied with a complex scalar """
        v = np.array([1.0, 0.0 + 2.1j, 3.8, 4.6 + 3.6j])
        s = 3.5  # This is now a real input
        sVec = pym.Signal("vec", v.copy())
        sScalar = pym.Signal("scalar", s)

        sRes = pym.Signal("result")

        mod = pym.MathGeneral([sVec, sScalar], sRes, expression="inp0*inp1")

        mod.response()

        # Check result
        self.assertIsInstance(sRes.state, type(sVec.state))
        self.assertTrue(np.allclose(sRes.state, v * s))

        dr = np.array([1.0 + 0.1j, -2.0 - 2.5j, 0.0 + 0.8j, -4.0 - 2.2j])
        sRes.sensitivity = dr.copy()

        mod.sensitivity()

        # Check sensitivity types
        self.assertIsInstance(sScalar.sensitivity, type(sScalar.state))
        self.assertIsInstance(sVec.sensitivity, type(sVec.state))
        self.assertEqual(sVec.sensitivity.dtype, sVec.state.dtype)

        # Check results
        self.assertAlmostEqual(sScalar.sensitivity, np.real(dr @ v))
        self.assertTrue(np.allclose(sVec.sensitivity, dr * s))

    def testVec_npScalar_complex(self):
        """ Test functionality where a vector is multiplied with a scalar """
        v = np.array([1.0, 0.0 + 2.1j, 3.8, 4.6 + 3.6j])
        s = np.array(3.5 + 3.8j)
        sVec = pym.Signal("vec", v.copy())
        sScalar = pym.Signal("numpy_scalar", s.copy())

        sRes = pym.Signal("result")

        mod = pym.MathGeneral([sVec, sScalar], sRes, expression="inp0*inp1")

        mod.response()

        # Check result
        self.assertIsInstance(sRes.state, type(sVec.state))
        self.assertTrue(np.allclose(sRes.state, v*s))

        dr = np.array([1.0 + 0.1j, -2.0 - 2.5j, 0.0 + 0.8j, -4.0 - 2.2j])
        sRes.sensitivity = dr.copy()

        mod.sensitivity()

        # Check sensitivity types
        self.assertIsInstance(sScalar.sensitivity, type(sScalar.state))
        self.assertIsInstance(sVec.sensitivity, type(sVec.state))

        # Check results
        self.assertAlmostEqual(sScalar.sensitivity, dr@v)
        self.assertTrue(np.allclose(sVec.sensitivity, dr*s))

    def testVec_npScalar_complex_real_input(self):
        """ Test functionality where a vector is multiplied with a scalar """
        v = np.array([1.0, 0.0 + 2.1j, 3.8, 4.6 + 3.6j])
        s = np.array(3.5)
        sVec = pym.Signal("vec", v.copy())
        sScalar = pym.Signal("numpy_scalar", s.copy())

        sRes = pym.Signal("result")

        mod = pym.MathGeneral([sVec, sScalar], sRes, expression="inp0*inp1")

        mod.response()

        # Check result
        self.assertIsInstance(sRes.state, type(sVec.state))
        self.assertTrue(np.allclose(sRes.state, v*s))

        dr = np.array([1.0 + 0.1j, -2.0 - 2.5j, 0.0 + 0.8j, -4.0 - 2.2j])
        sRes.sensitivity = dr.copy()

        mod.sensitivity()

        # Check sensitivity types
        self.assertIsInstance(sScalar.sensitivity, type(sScalar.state))
        self.assertIsInstance(sVec.sensitivity, type(sVec.state))

        # Check results
        self.assertAlmostEqual(sScalar.sensitivity, np.real(v@dr))
        self.assertTrue(np.allclose(sVec.sensitivity, s*dr))

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-7, atol=1e-5))
        pym.finite_difference(mod, test_fn=tfn)


if __name__ == '__main__':
    unittest.main()
