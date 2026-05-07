import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym


class TestConcat:
    def test_concatenate(self):
        s1 = pym.Signal('sig1', 0.0)
        s2 = pym.Signal('sig2', np.array([1.0, 2.0, 3.0, 4.0]))
        s3 = pym.Signal('sig3', np.float32(5.0))
        s4 = pym.Signal('sig4', np.array([[6.0, 7.0]]))
        s5 = pym.Signal('sig5', np.array(8.0))

        s_out = pym.Signal('out')
        m = pym.Concatenate().connect([s1, s2, s3, s4, s5], s_out)

        npt.assert_equal(s_out.state, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))

        # Set the state of the concatenated signal
        s_out.sensitivity = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.1])
        m.sensitivity()

        assert s1.sensitivity == 8.0
        npt.assert_equal(s2.sensitivity, np.array([7.0, 6.0, 5.0, 4.0]))
        assert s3.sensitivity == np.float32(3.0)
        npt.assert_equal(s4.sensitivity, np.array([[2.0, 1.0]]))
        npt.assert_equal(s5.sensitivity, np.array(0.1))

        for s in m.sig_in:
            assert isinstance(s.sensitivity, type(s.state))
            if hasattr(s.state, 'shape'):
                assert s.sensitivity.shape == s.state.shape

        # Should return Nones
        m.reset()
        s_out.sensitivity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        m.sensitivity()

        for s in m.sig_in:
            assert s.sensitivity is None

    def test_slicing_error(self):
        # May 7, 2026. Does not work with Numpy 2.4.4, works in 2.3.3
        # Error in two places:
        #  - Concatenate does not return same type as input signal for sensitivities <-- fixed in Concatenate update
        #  - Sliced signal adding sensitivity of shape (1, ) to shape () does not work.
        with pym.Network() as fn:
            s_inp = pym.Signal('s_out', np.random.rand(2, ))

            s_concat = pym.Concatenate()(s_inp[0], s_inp[1])
            s_agg = pym.PNorm(p=40)(s_concat)
            s_g = pym.Scaling(scaling=10., maxval=0.05)(s_agg)

        s_g.sensitivity = 1.0
        fn.sensitivity()

if __name__ == '__main__':
    pytest.main([__file__])
