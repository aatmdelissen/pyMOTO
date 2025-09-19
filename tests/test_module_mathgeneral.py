import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym
import copy


def make_vec(n):
    return np.random.rand(n)


def make_mat(m, n=None):
    if n is None:
        n = m
    return np.random.rand(m, n)


class TestMathGeneral:
    np.random.seed(0)

    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-6, atol=1e-15)

    @pytest.mark.parametrize('b', [3.5, np.array(3.5), 3.5 + 3.8j], ids=['float', 'np_scalar', 'complex_float'])
    @pytest.mark.parametrize('a', 
                             [make_vec(4), make_vec(5) + 1j*make_vec(5), make_mat(3)], 
                             ids=['vector', 'complex_vector', 'matrix'])
    def test_vec_scalar_multiply(self, a, b):
        """ Test functionality where a vector is multiplied with a scalar """
        sVec = pym.Signal("a", copy.deepcopy(a))
        sScalar = pym.Signal("b", copy.deepcopy(b))

        mod = pym.MathGeneral("inp0*inp1")
        sRes = mod(sVec, sScalar)
        sRes.tag = 'result'

        # Check result
        assert isinstance(sRes.state, type(a*b))
        npt.assert_equal(sRes.state, a*b)
        assert sRes.state.shape == a.shape

        sRes.sensitivity = np.random.rand(*np.shape(sRes.state))
        if np.iscomplexobj(sRes.state):
            sRes.sensitivity = sRes.sensitivity + 1j*np.random.rand(*np.shape(sRes.state))

        mod.sensitivity()

        # Check sensitivity types
        assert isinstance(sVec.sensitivity, type(sVec.state))
        assert isinstance(sScalar.sensitivity, type(sScalar.state))

        # Check results
        sens_scalar = np.sum(sRes.sensitivity * a)
        npt.assert_allclose(sScalar.sensitivity, np.real(sens_scalar) if np.isrealobj(sScalar.state) else sens_scalar)
        sens_vec = sRes.sensitivity * b
        npt.assert_allclose(sVec.sensitivity, np.real(sens_vec) if np.isrealobj(sVec.state) else sens_vec)

        # Check finite difference
        pym.finite_difference(tosig=sRes, test_fn=self.fd_testfn)

    def test_numpy_arrays_without_broadcasting(self):
        """ Check if broadcasting works for sensitivity calculation """
        np.random.seed(0)
        sv1 = pym.Signal("v1", np.random.rand(15))
        sv2 = pym.Signal("v2", np.random.rand(15))

        sRes = pym.MathGeneral("inp0*inp1")(sv1, sv2)
        sRes.tag = 'v1*v2'

        # Check value of response
        assert sRes.state.shape == (15,)
        npt.assert_allclose(sRes.state, sv1.state * sv2.state)

        # Check sensitivities
        pym.finite_difference(tosig=sRes, test_fn=self.fd_testfn)

    def test_numpy_arrays_with_broadcasting(self):
        """ Check if broadcasting works for sensitivity calculation """
        np.random.seed(0)
        sv1 = pym.Signal("v1", np.random.rand(15))
        sv2 = pym.Signal("v2", np.random.rand(2, 15))
        sv3 = pym.Signal("v3", np.random.rand(2, 2, 15))
        s_scalar = pym.Signal("c", 3.5)

        sRes = pym.MathGeneral("inp0*inp1*inp2*inp3")(sv1, sv2, sv3, s_scalar)
        sRes.tag = 'v1*v2*v3*c'

        # Check value of response
        assert sRes.state.shape == (2, 2, 15)
        npt.assert_allclose(sRes.state, sv1.state * sv2.state * sv3.state * s_scalar.state)

        # Check sensitivities
        pym.finite_difference(tosig=sRes, test_fn=self.fd_testfn)

    def test_numpy_arrays_with_broadcasting1(self):
        """ Broadcast with singleton axes """
        np.random.seed(0)
        sv1 = pym.Signal("v1", np.random.rand(36, 1, 15))
        sv2 = pym.Signal("v2", np.random.rand(2, 1, 1, 15))
        sv3 = pym.Signal("v3", np.random.rand(2, 1, 2, 15))
        s_scalar = pym.Signal("c", 3.5)

        sRes = pym.MathGeneral("inp0*inp1*inp2*inp3")(sv1, sv2, sv3, s_scalar)
        sRes.tag = 'v1*v2*v3*c'

        # Check value of response
        assert sRes.state.shape == (2, 36, 2, 15)
        npt.assert_allclose(sRes.state, sv1.state * sv2.state * sv3.state * s_scalar.state)

        # Check sensitivities
        pym.finite_difference(tosig=sRes, test_fn=self.fd_testfn)

    def test_numpy_arrays_with_one_constant(self):
        """ Broadcast with singleton axes """
        np.random.seed(0)
        sv1 = pym.Signal("v1", np.random.rand(36, 1, 15))
        sv2 = pym.Signal("v2", np.random.rand(2, 1, 1, 15))
        v3 = np.random.rand(2, 1, 2, 15)
        s_scalar = pym.Signal("c", 3.5)

        sRes = pym.MathGeneral("inp0*inp1*inp2*inp3")(sv1, sv2, v3, s_scalar)
        sRes.tag = 'v1*v2*v3*c'

        # Check value of response
        assert sRes.state.shape == (2, 36, 2, 15)
        npt.assert_allclose(sRes.state, sv1.state * sv2.state * v3 * s_scalar.state)

        # Check sensitivities
        pym.finite_difference(tosig=sRes, test_fn=self.fd_testfn)

    def test_with_only_constant(self):
        val = pym.MathGeneral("sqrt(inp0)")(np.array([1,2,3]))
        npt.assert_allclose(val, np.sqrt(np.array([1,2,3])))


if __name__ == '__main__':
    pytest.main([__file__])
