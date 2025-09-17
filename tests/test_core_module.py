import pytest
import numpy.testing as npt
import pymoto as pym
import numpy as np


class TestModule:
    def test_initialize1_with_signal(self):
        a = pym.Signal('x_in', 1.0)
        pytest.raises(TypeError, pym.Module, msg="Can't instantiate the abstract base class without implementation")

        class MyMod(pym.Module):
            def __call__(self, x):
                return x*2

        m = MyMod()
        b = m(a)
        assert isinstance(b, pym.Signal)
        assert len(m.sig_in) == 1, "Should have 1 input"
        assert len(m.sig_out) == 1, "Should have 1 output"
        assert m.sig_in[0] == a, "Input should be a"
        assert m.sig_out[0] == b, "Output should be b"
        assert b.state == a.state * 2

    def test_initialize1_with_value(self):
        a = 1.0

        class MyMod(pym.Module):
            def __call__(self, x):
                return x*2

        m = MyMod()
        b = m(a)
        assert isinstance(b, type(a))
        assert m.sig_in is None
        assert m.sig_out is None
        assert b == a*2

    def test_initialize2_homogeneous_with_signals(self):
        a = pym.Signal('x1', 1.0)
        b = pym.Signal('x2', 2.0)
        c = pym.Signal('x3', 3.0)

        class MyMod(pym.Module):
            def __call__(self, x, y, z):
                return x+y, z+x

        m = MyMod()
        d, e = m(a, b, c)

        assert len(m.sig_in) == 3
        assert len(m.sig_out) == 2
        assert m.sig_in[0] == a
        assert m.sig_in[1] == b
        assert m.sig_in[2] == c
        assert m.sig_out[0] == d
        assert m.sig_out[1] == e
        assert m.sig_out[0].state == a.state + b.state
        assert m.sig_out[1].state == c.state + a.state

    def test_initialize2_homogeneous_with_values(self):
        a = 1.0
        b = 2.0
        c = 3.0

        class MyMod(pym.Module):
            def __call__(self, x, y, z):
                return x+y, z+x

        m = MyMod()
        d, e = m(a, b, c)

        assert m.sig_in is None
        assert m.sig_out is None
        assert d == a + b
        assert e == c + a

        pytest.raises(RuntimeError, m.sensitivity)

    def test_initialize2_heterogeneous(self):
        a = pym.Signal('x1', 1.0)
        b = 2.0
        c = 3.0

        class MyMod(pym.Module):
            def __call__(self, x, y, z):
                return x+y, z+x

        m = MyMod()
        d, e = m(a, b, c)

        assert len(m.sig_in) == 3
        assert len(m.sig_out) == 2
        assert m.sig_in[0] == a
        assert m.sig_in[1] == b
        assert m.sig_in[2] == c
        assert m.sig_out[0] == d
        assert m.sig_out[1] == e
        assert m.sig_out[0].state == a.state + b
        assert m.sig_out[1].state == c + a.state

    def test_response_and_sens(self):
        class TwoInTwoOut(pym.Module):
            def __init__(self, argument):
                self.prepared = argument
                self.internalstate = 0
                self.didsensitivity = False

            def __call__(self, a, b):
                self.internalstate += 1
                return a * b, a + b

            def _sensitivity(self, dc, dd):
                self.didsensitivity = True
                if dc is None:
                    dc = 0.0
                if dd is None:
                    dd = 0.0
                a, b = [s.state for s in self.sig_in]
                return b * dc + dd, a * dc + dd

            def _reset(self):
                self.internalstate = 0
                self.didsensitivity = False

        sa = pym.Signal('a', 2.5)
        sb = pym.Signal('b', 3.5)

        m = TwoInTwoOut('foo')
        assert m.prepared == 'foo', "Check if the preparation has been executed"
        sc, sd = m(sa, sb)
        assert m.internalstate == 1, "Check if response has been called"

        m.response()
        assert m.internalstate == 2, "Check if response has been called again"

        m.sensitivity()
        assert not m.didsensitivity, "Sensitivity should not have been called, since output sensitivities are None"

        # one output sensitivity
        sc.sensitivity = 1.0
        m.sensitivity()
        assert m.didsensitivity, "Sensitivity should have been called"
        assert sa.sensitivity == 1.0 * sb.state  # c = a * b
        assert sb.sensitivity == 1.0 * sa.state

        # Reset
        m.reset()
        assert all(s.sensitivity is None for s in [sa, sb, sc, sd])
        assert not m.internalstate, "Check if reset has worked"
        assert not m.didsensitivity, "Check if reset has worked"

        # other output sensitivity
        sd.sensitivity = 1.0
        m.sensitivity()
        assert m.didsensitivity, "Sensitivity should have been called"
        assert sa.sensitivity == 1.0  # d = a + b
        assert sb.sensitivity == 1.0

        # Both sensitivities
        m.reset()
        sc.sensitivity = 1.0
        sd.sensitivity = 1.0
        m.sensitivity()
        assert m.didsensitivity, "Sensitivity should have been called"
        assert sa.sensitivity == 4.5
        assert sb.sensitivity == 3.5

    def test_source_module(self):
        class SourceMod(pym.Module):
            def __init__(self, value=3.14):
                self.value = value

            def __call__(self):
                return self.value

        # No arguments
        m = SourceMod()
        s = m()

        assert isinstance(s, pym.Signal)
        assert s.state == 3.14

        # Change argument
        m.value = 4.5
        m.response()
        assert s.state == 4.5

        # With init argument
        m1 = SourceMod(5.9)
        s1 = m1()
        assert s1.state == 5.9

    def test_sink_module_wsignal(self):
        class SinkMod(pym.Module):
            def __call__(self, in1):
                self.got_in1 = in1

            def _sensitivity(self):
                self.did_sens = True
                return 2.15

        a = pym.Signal('x_in', 1.256)
        bl = SinkMod()
        bl(a)
        assert bl.got_in1 == 1.256, "State variable passed to _reponse function"

        bl.sensitivity()
        assert bl.did_sens, "Check if _sensitivity did run"
        assert a.sensitivity == 2.15, "After running first sensitivity"
        bl.sensitivity()
        assert a.sensitivity == 2.15 + 2.15, "After running second sensitivity"
        bl.reset()
        assert a.sensitivity is None, "After resetting module"
        bl.sensitivity()
        assert a.sensitivity == 2.15, "First sensitivity run after reset"

    def test_sink_module_wvalue(self):
        class SinkMod(pym.Module):
            def __call__(self, in1):
                self.got_in1 = in1

            def _sensitivity(self):
                self.did_sens = True
                return 2.15

        a = 1.256
        m = SinkMod()
        m(a)
        assert m.got_in1 == 1.256, "State variable passed to _reponse function"

        pytest.raises(RuntimeError, m.sensitivity)

    def test_wrong_number_of_inputs(self):
        class WrongResponse(pym.Module):
            def __call__(self, a):
                return a * 2.0, a * 3.0  # Two returns

        sa = pym.Signal('a', 2.5)
        sb = pym.Signal()

        m = WrongResponse()  # One output signal
        pytest.raises(TypeError, m, sa, sb, msg="Number of out-signals should match number of returns in its response")

    def test_module_noresponse(self):
        class WrongResponse(pym.Module):
            def __wrong_call__(self, a):
                return a * 2.0, a * 3.0  # Two returns
        pytest.raises(TypeError, WrongResponse, msg="__call__ should be implemented")

    def test_faulty_sensitivity_definition(self):
        """ To check error messages """
        class WrongSensitivity(pym.Module):
            def __call__(self, a, b):
                return a * b

            def _sensitivity(self, dc):
                self.did_sensitivity = True
                a, b = self.get_input_states()
                return b * dc  # ERROR: Only returns one sensitivity

            def _reset(self):
                raise RuntimeError("An error has occurred")

        sa = pym.Signal('a', 2.5)
        sb = pym.Signal('b', 3.5)

        m = WrongSensitivity().connect([sa, sb])

        sc = m.sig_out[0]  # Two inputs -> expects two sensitivities returned

        m.sensitivity()  # First test with None as sensitivity, no error as sensitivity is not run
        assert not hasattr(m, "did_sensitivity")

        sc.sensitivity = 1.0
        pytest.raises(TypeError, m.sensitivity)  # Error as two sensitivites are expected
        try:
            m.sensitivity()
        except TypeError as e:
            print("Error when running m.sensitivity(): -------\n\n")
            print(e)
        assert m.did_sensitivity

        pytest.raises(RuntimeError, m.reset)
        try:
            m.reset()
        except RuntimeError as e:
            print("Error when running m.reset(): -------\n\n")
            print(e)

    def test_reset_error(self):
        class ErrModule(pym.Module):
            def __call__(self, a, b):
                return a * b

            def _sensitivity(self, dc):
                raise ValueError("some error in calculation")

        sa = pym.Signal('a', 2.5)
        sb = pym.Signal('b', 3.5)
        m1 = ErrModule().connect([sa, sb])
        sc = m1.sig_out[0]

        sc.sensitivity = 1.0
        pytest.raises(ValueError, m1.sensitivity)
        try:
            m1.sensitivity()
        except ValueError as e:
            print("Error when running m.sensitivity(): -------\n\n")
            print(e)

    def test_identical_sensitivity(self):
        """ Check to see if identically variables returned in _sensitivity are handled correctly"""
        class Add(pym.Module):
            def __call__(self, A, B):
                return A + B

            def _sensitivity(self, dC):
                return dC, dC

        s_A = pym.Signal('A', np.array([1, 2, 3]))
        s_B = pym.Signal('B', np.array([4, 5, 6]))
        m = Add()
        s_C = m(s_A, s_B)
        s_C.sensitivity = np.array([1, 1, 1])
        m.sensitivity()
        assert not np.may_share_memory(s_A.sensitivity, s_B.sensitivity)  # May not share same reference
        assert not np.may_share_memory(s_A.sensitivity, s_C.sensitivity)

    def test_keyword_arguments(self):
        class VarArg(pym.Module):
            def __call__(self, A, B=1.0):
                return A + B

            def _sensitivity(self, dC):
                return dC if self.n_in == 1 else (dC, dC)

        s_A = pym.Signal('A', np.array([1, 2, 3]))
        s_B = pym.Signal('B', np.array([4, 5, 6]))

        # With one argument
        m = VarArg()
        s_C = m(s_A)
        npt.assert_equal(s_C.state, np.array([2, 3, 4]))
        s_C.sensitivity = np.array([1, 1, 1])
        m.sensitivity()
        npt.assert_equal(s_A.sensitivity, np.array([1, 1, 1]))
        assert s_B.sensitivity is None
        assert not np.may_share_memory(s_A.sensitivity, s_B.sensitivity)

        m.reset()

        # With two arguments
        m = VarArg()
        s_C = m(s_A, s_B)
        npt.assert_equal(s_C.state, np.array([5, 7, 9]))

        s_C.sensitivity = np.array([1, 1, 1])
        m.sensitivity()
        npt.assert_equal(s_A.sensitivity, np.array([1, 1, 1]))
        npt.assert_equal(s_B.sensitivity, np.array([1, 1, 1]))
        assert not np.may_share_memory(s_A.sensitivity, s_B.sensitivity)
        assert not np.may_share_memory(s_A.sensitivity, s_C.sensitivity)

    def test_two_keyword_arguments(self):
        # TODO: Keyword arguments are not supported (yet)
        class Var2Arg(pym.Module):
            def __call__(self, A, B=1.0, C=2.0):
                return C * (A + B)

            def _sensitivity(self, dC):
                return dC if self.n_in == 1 else (dC, dC)

        s_A = pym.Signal('A', np.array([1., 2., 3.]))
        s_B = pym.Signal('B', np.array([4., 5., 6.]))

        # With B as argument
        m = Var2Arg()
        pytest.raises(TypeError, m, s_A, B=s_B)
       

    def test_reconnect_module(self):
        class MyMod(pym.Module):
            def __call__(self, A, B):
                return A + B

            def _sensitivity(self, dC):
                return dC, dC

        s_A = pym.Signal('A', np.array([1, 2, 3]))
        s_B = pym.Signal('B', np.array([4, 5, 6]))
        s_C = pym.Signal('A', np.array([1, 2]))
        s_D = pym.Signal('B', np.array([4, 5]))
        m = MyMod()
        s_E = m(s_A, s_B)
        pytest.raises(RuntimeError, m, s_C, s_D)  # Error when trying to connect module twice
        # s_F = m(s_C, s_D)
        npt.assert_equal(s_E.state, np.array([5, 7, 9]))
        # npt.assert_equal(s_F.state, np.array([5, 7]))

    def test_connect_module(self):
        class MyMod(pym.Module):
            def __call__(self, A, B):
                return A + B

            def _sensitivity(self, dC):
                return dC, dC

        m = MyMod().connect([pym.Signal('A', np.array([1, 2, 3])), pym.Signal('B', np.array([4, 5, 6]))])
        assert isinstance(m, MyMod)
        assert len(m.sig_out) == 1
        npt.assert_equal(m.sig_out[0].state, np.array([5, 7, 9]))
