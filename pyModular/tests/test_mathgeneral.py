from unittest import TestCase
import numpy as np
import pyModular as pym


class TestMath(TestCase):

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
        self.assertIsInstance(sScalar.sensitivity, type(sScalar.sensitivity))
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
        self.assertIsInstance(sScalar.sensitivity, type(sScalar.sensitivity))
        self.assertIsInstance(sVec.sensitivity, type(sVec.state))

        # Check results
        self.assertAlmostEqual(sScalar.sensitivity, 1.0*1.0 - 0.0*2.0 + 3.8*0.0 - 4.6*4.0)
        self.assertTrue(np.allclose(sVec.sensitivity, np.array([1.0*3.5, -2.0*3.5, 0.0*3.5, -4.0*3.5])))
