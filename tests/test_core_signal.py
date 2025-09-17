import pytest
import numpy.testing as npt
import pymoto as pym
import numpy as np


class TestSignal:
    def test_initialize(self):
        a = pym.Signal('foo')
        assert a.tag == 'foo', "Initialize tag"
        assert a.state is None, "Initialize state to None"
        assert a.sensitivity is None, "Initialize sensitivity to None"


    def test_state(self):
        a = pym.Signal('foo')
        a.state = 1.0
        assert a.state == 1.0, "Set state to scalar"

        val = np.array([1.0, 2.0, 3.0])
        a.state = val
        npt.assert_equal(a.state, val), "Set state to array"
        np.shares_memory(a.state, val)

        b = pym.Signal('foo', np.array([5.0, 6.0]))
        assert b.tag == 'foo', "Set tag from init with state"
        npt.assert_equal(b.state, np.array([5.0, 6.0])), "Set state from init"

    def test_sensitivity(self):
        a = pym.Signal('foo')
        a.sensitivity = 2.0
        assert a.sensitivity == 2.0, "Set initial sensitivity to scalar"

        a.add_sensitivity(3.0)
        assert a.sensitivity == 5.0, "Add scalar sensitivity"

        a.sensitivity = 1.0
        assert a.sensitivity == 1.0, "Rewrite sensitivity by set_sens"

        a.reset(keep_alloc=True)
        assert a.sensitivity == 0.0, "Reset while keeping memory allocation"

        a.reset()
        assert a.sensitivity is None, "Reset sensitivity"

        a.add_sensitivity(np.array([1.1, 2.2, 3.3]))
        npt.assert_equal(a.sensitivity, np.array([1.1, 2.2, 3.3])), "Set initial sensitivity by add_sensitivity"

        a.add_sensitivity(None)
        npt.assert_equal(a.sensitivity, np.array([1.1, 2.2, 3.3])), "After adding None by add_sensitivity"

        b = pym.Signal('foo', np.array([5.0, 6.0]), np.array([7.0, 8.0]))
        assert b.tag == 'foo', "Set tag from init with state and sensitivity"
        npt.assert_equal(b.state, np.array([5.0, 6.0])), "Set state from init and sensitivity"
        npt.assert_equal(b.sensitivity, np.array([7.0, 8.0])), "Set sensitivity from init and sensitivity"

        c = pym.Signal('bar', sensitivity=np.array([7.0, 8.0]))
        assert c.tag == 'bar', "Set tag from init with sensitivity"
        npt.assert_equal(c.sensitivity, np.array([7.0, 8.0])), "Set sensitivity from init with sensitivity"

    def test_make_signals(self):
        d = pym.make_signals('a', 'b', 'c')
        assert isinstance(d['a'], pym.Signal)
        assert isinstance(d['b'], pym.Signal)
        assert isinstance(d['c'], pym.Signal)
        assert d['a'].tag == 'a'
        assert d['b'].tag == 'b'
        assert d['c'].tag == 'c'

    def test_add_sensitivity_errors(self):
        a = pym.Signal('foo')
        a.sensitivity = np.array([1.0, 2.0, 3.0])
        # Add wrong type
        pytest.raises(TypeError, a.add_sensitivity, "cannot add a string")
        # a.add_sensitivity("cannot add a string")

        # Add wrong value
        pytest.raises(ValueError, a.add_sensitivity, np.random.rand(3, 3))
        # a.add_sensitivity(np.random.rand(3, 3))

        # Adding complex array to real array does not give problems
        b = pym.Signal('real', np.random.rand(3))
        b.add_sensitivity(np.random.rand(3) + 1j*np.random.rand(3))

        c = pym.Signal('integer', 1)
        c.add_sensitivity(1.234)
        c.add_sensitivity(np.array(1.345))
        c.add_sensitivity(np.array([1.3344]))
        # c.add_sensitivity(np.array([[394]]))
        pytest.raises(ValueError, c.add_sensitivity, np.array([[23454]]))  # Cannot add this shape to existing array

        d = pym.Signal('integer', 1.0)
        d.add_sensitivity(np.array([[23454]]))  # But it can be added to a float

        # Type which doesnt have +=
        class MyObj:
            def __init__(self, val):
                self.val = val
        e = pym.Signal('foo')
        e.add_sensitivity(MyObj(1.3))
        # e.add_sensitivity(MyObj(3.4))
        pytest.raises(TypeError, e.add_sensitivity, MyObj(3.4))

    def test_reset_errors(self):
        a = pym.Signal('floatingpoint')
        a.add_sensitivity(1.3)
        a.reset(True)

        # With an object that doesnt have [] and *=
        class MyObj:
            def __init__(self, val):
                self.val = val
        b = pym.Signal('foo')
        b.add_sensitivity(MyObj(1.3)), pytest.warns(RuntimeWarning, b.reset, True)
        # b.add_sensitivity(MyObj(1.3)), b.reset(True)  # Gives a warning, and just replaced by None

    def test_reference_copy(self):
        a = pym.Signal('original', 1.0)
        b = a

        a.state = 2.0
        a.sensitivity = 3.0
        assert a.state == b.state
        assert b.sensitivity == 3.0


class TestSignalSlice:
    def test_slice_1d(self):

        sx = pym.Signal('x')
        sx.state = np.random.rand(100)
        sx.state[1] = 0.314
        assert sx[0].state == sx.state[0]
        assert np.allclose(sx[0:5].state, sx.state[0:5])
        sx[0].state = 0.1
        assert sx[0].state == 0.1
        assert sx.state[0] == 0.1
        assert sx[1].state == 0.314

        sx[0].state += 0.1
        assert sx[0].state == 0.2
        assert sx.state[0] == 0.2

        sx[4:9].state = 0.1
        assert np.allclose(sx[4:9].state, 0.1)
        assert np.allclose(sx.state[4:9], 0.1)

        sx[4:9].state += 0.1
        assert np.allclose(sx[4:9].state, 0.2)
        assert np.allclose(sx.state[4:9], 0.2)

        assert sx[0].sensitivity is None
        assert sx[1:34, 53, 123:56, [2, 34, 5]].sensitivity is None

        sx[1].add_sensitivity(1.0)
        pytest.raises(ValueError, sx[1].add_sensitivity, np.array([1, 2, 34]))
        assert sx.state.size == sx.sensitivity.size
        assert sx[1].state.size == sx[1].sensitivity.size
        assert sx[1].sensitivity == 1.0
        assert sx[0].sensitivity == 0.0

        sx[0].add_sensitivity(2.0)
        assert sx[0].sensitivity == 2.0
        assert sx.sensitivity[0] == 2.0
        assert sx[1].sensitivity == 1.0

        sx[0].reset()
        assert sx[0].sensitivity == 0.0
        assert sx[1].sensitivity == 1.0

        sx[4:8].add_sensitivity(3.0)
        assert sx[0].sensitivity == 0.0
        assert sx[1].sensitivity == 1.0
        assert np.allclose(sx[4:8].sensitivity, 3.0)
        assert np.allclose(sx.sensitivity[4:8], 3.0)
        assert np.allclose(sx[9].sensitivity, 0.0)

        sx[9:12].add_sensitivity(np.array([4, 5, 6]))
        pytest.raises(ValueError, sx[9:12].add_sensitivity, np.array([4, 5, 6, 4]))
        assert np.allclose(sx[9:12].sensitivity, np.array([4, 5, 6]))
        sx[9:11].reset()
        assert np.allclose(sx[9:11].sensitivity, 0)
        assert np.allclose(sx.sensitivity[9:11], 0)
        assert sx[11].sensitivity == 6.0

    def test_slice_error(self):
        class MyObj:
            def __init__(self, val):
                self.val = val

        def call_state(s):
            return s.state

        def call_sens(s):
            return s.sensitivity

        # Object that cannot be sliced
        a = pym.Signal("myobj", state=MyObj(1.23), sensitivity=MyObj(1.34))
        # call_state(a[2]) # Empty state
        # call_sens(a[2])
        pytest.raises(TypeError, call_state, a[2])
        pytest.raises(TypeError, call_sens, a[2])

        # Too many dimensions
        b = pym.Signal("2dim", state=np.random.rand(10, 10), sensitivity=np.random.rand(10, 10))
        # call_state(b[2,3,4]) # Too many dimensions
        # call_sens(b[2,3,4])
        pytest.raises(IndexError, call_state, b[2, 3, 4])
        pytest.raises(IndexError, call_sens, b[2, 3, 4])
        # call_state(b[np.array([1,2,493]), 1]) # Out of range
        # call_sens(b[np.array([1,2,493]), 1])
        pytest.raises(IndexError, call_state, b[np.array([1, 2, 493]), 1])
        pytest.raises(IndexError, call_sens, b[np.array([1, 2, 493]), 1])

    def test_sensitivity_set_error(self):
        s = pym.Signal("empty")

        def set_sens(s, val):
            s.sensitivity = val

        # set_sens(s[2], 3.4) # Cannot set when state is None
        pytest.raises(TypeError, set_sens, s[2], 3.4)

        class MyObj:
            def __init__(self, val):
                self.val = val
        a = pym.Signal("obj", MyObj(1.3))
        # set_sens(a[3], 3.4) # Cannot zero-initialize this object
        pytest.raises(TypeError, set_sens, a[3], 3.4)

    def test_slice_2d(self):
        s = pym.Signal("2D_vals", np.random.rand(10, 10))

        s[0, 4].state = 0.4
        assert s[0, 4].state == 0.4
        assert s.state[0, 4] == 0.4

        s[:, 3].state = 0.8
        assert np.all(s[:, 3].state == 0.8)
        assert np.all(s.state[:, 3] == 0.8)

        s[0, 2:8].sensitivity = 1.0
        assert s.sensitivity.shape == s.state.shape
        assert np.all(s.sensitivity[0, 2:8] == 1.0)
        assert s.sensitivity[0, 0] == 0.0
        assert s.state[0, 4] == 0.4  # Must be still the same as previously
        assert np.all(s.state[:, 3] == 0.8)

        s[0, 2:8].sensitivity = 0.0
        assert np.all(s.sensitivity == 0.0)

        # Test add_sensitivity
        add_arr = np.random.rand(6)
        s[0, 2:8].add_sensitivity(add_arr)
        assert np.all(s.sensitivity[0, 2:8] == add_arr)
        s[0, 2:8].add_sensitivity(add_arr)
        assert np.all(s.sensitivity[0, 2:8] == 2*add_arr)

        # Test reset
        s[0, 2:8].reset(keep_alloc=False)
        assert np.all(s.sensitivity[0, 2:8] == 0)

        s[0, 2:8].add_sensitivity(add_arr)
        s[0, 2:8].reset(keep_alloc=True)
        assert np.all(s.sensitivity[0, 2:8] == 0)

        s[0, 2:8].add_sensitivity(add_arr)
        s[0, 2:8].add_sensitivity(None)
        assert np.all(s.sensitivity[0, 2:8] == add_arr)

    def test_numpy_slice(self):
        # To test bug where sensitivities are not set correctly through the sliced signal
        sx = pym.Signal('x', np.random.rand(15))

        # Test state setter
        sx[np.arange(9, 12)].state = 1.5
        assert np.allclose(sx.state[np.arange(9, 12)], 1.5)

        sx[np.arange(9, 12)].state[:] = 2.0

        # Test with numpy slice
        sx_sliced = sx[np.arange(9, 12)]
        sx_sliced.add_sensitivity(np.array([4, 5, 6]))
        pytest.raises(ValueError, sx_sliced.add_sensitivity, np.array([4, 5, 6, 4]))
        assert np.allclose(sx_sliced.sensitivity, np.array([4, 5, 6]))
        sx[np.arange(9, 11)].reset(keep_alloc=True)
        assert np.allclose(sx[np.arange(9, 11)].sensitivity, 0)
        assert np.allclose(sx.sensitivity[np.arange(9, 11)], 0)
        assert sx[11].sensitivity == 6.0
