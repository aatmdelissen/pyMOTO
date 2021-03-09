from unittest import TestCase
import pymodular as pym
import numpy as np


class TestSignal(TestCase):
    def test_initialize(self):
        a = pym.Signal('foo')
        self.assertEqual(a.tag, 'foo', msg="Initialize tag")
        self.assertIsNone(a.state, msg="Initialize state to None")
        self.assertIsNone(a.state, msg="Initialize sensitivity to None")

    def test_state(self):
        a = pym.Signal('foo')
        a.state = 1.0
        self.assertEqual(a.state, 1.0, msg="Set state to scalar")

        a.state = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.allclose(a.state, np.array([1.0, 2.0, 3.0])), msg="Set state to array")

        b = pym.Signal('foo', np.array([5.0, 6.0]))
        self.assertEqual(b.tag, 'foo', msg="Set tag from init with state")
        self.assertTrue(np.allclose(b.state, np.array([5.0, 6.0])), msg="Set state from init")

    def test_sensitivity(self):
        a = pym.Signal('foo')
        a.sensitivity = 2.0
        self.assertEqual(a.sensitivity, 2.0, msg="Set initial sensitivity to scalar")

        a.add_sensitivity(3.0)
        self.assertEqual(a.sensitivity, 5.0, msg="Add scalar sensitivity")

        a.sensitivity = 1.0
        self.assertEqual(a.sensitivity, 1.0, msg="Rewrite sensitivity by set_sens")

        a.reset(keep_alloc=True)
        self.assertEqual(a.sensitivity, 0.0, msg="Reset while keeping memory allocation")

        a.reset()
        self.assertEqual(a.sensitivity, None, msg="Reset sensitivity")

        a.add_sensitivity(np.array([1.1, 2.2, 3.3]))
        self.assertTrue(np.allclose(a.sensitivity, np.array([1.1, 2.2, 3.3])), msg="Set initial sensitivity by add_sensitivity")

        a.add_sensitivity(None)
        self.assertTrue(np.allclose(a.sensitivity, np.array([1.1, 2.2, 3.3])), msg="After adding None by add_sensitivity")

        self.assertRaises(ValueError, a.add_sensitivity, np.random.rand(3, 3))

        b = pym.Signal('foo', np.array([5.0, 6.0]), np.array([7.0, 8.0]))
        self.assertEqual(b.tag, 'foo', msg="Set tag from init with state and sensitivity")
        self.assertTrue(np.allclose(b.state, np.array([5.0, 6.0])), msg="Set state from init and sensitivity")
        self.assertTrue(np.allclose(b.sensitivity, np.array([7.0, 8.0])), msg="Set sensitivity from init and sensitivity")

        c = pym.Signal('bar', sensitivity=np.array([7.0, 8.0]))
        self.assertEqual(c.tag, 'bar', msg="Set tag from init with sensitivity")
        self.assertTrue(np.allclose(c.sensitivity, np.array([7.0, 8.0])), msg="Set sensitivity from init with sensitivity")

    def test_make_signals(self):
        a, b, c = pym.make_signals('a', 'b', 'c')
        self.assertIsInstance(a, pym.Signal)
        self.assertIsInstance(b, pym.Signal)
        self.assertIsInstance(c, pym.Signal)
        self.assertEqual(a.tag, 'a')
        self.assertEqual(b.tag, 'b')
        self.assertEqual(c.tag, 'c')


class TestModule(TestCase):
    def test_initialize1(self):
        a = pym.Signal('x_in')
        b = pym.Signal('x_out')
        self.assertRaises(TypeError, pym.Module, a, b, msg="Can't instantiate the abstract base class without implementation")

        class MyMod(pym.Module):
            def _response(self, x):
                return x*2

        mod = MyMod(a, b)
        self.assertEqual(len(mod.sig_in), 1, msg="Should have 1 input")
        self.assertEqual(len(mod.sig_out), 1, msg="Should have 1 output")
        self.assertEqual(mod.sig_in[0], a, msg="Input should be a")
        self.assertEqual(mod.sig_out[0], b, msg="Output should be b")

    def test_initialize2(self):
        a = pym.Signal('x1')
        b = pym.Signal('x2')
        c = pym.Signal('x3')
        d = pym.Signal('y1')
        e = pym.Signal('y2')

        class MyMod(pym.Module):
            def _response(self, x, y, z):
                return x+y, y+x

        bl = MyMod([a, b, c], [d, e])

        self.assertEqual(len(bl.sig_in), 3)
        self.assertEqual(len(bl.sig_out), 2)
        self.assertEqual(bl.sig_in[0], a)
        self.assertEqual(bl.sig_in[1], b)
        self.assertEqual(bl.sig_in[2], c)
        self.assertEqual(bl.sig_out[0], d)
        self.assertEqual(bl.sig_out[1], e)

    def test_create(self):
        class FooMod(pym.Module):
            def _response(self, a_in):
                return a_in * 2

        print("test_create: ")
        pym.Module.print_children()
        a = pym.Signal('x_in')
        b = pym.Signal('x_out')
        bl = pym.Module.create('foomod', a, b)

        self.assertIsInstance(bl, pym.Module)
        self.assertIsInstance(bl, FooMod)

        self.assertEqual(len(bl.sig_in), 1)
        self.assertEqual(len(bl.sig_out), 1)
        self.assertEqual(bl.sig_in[0], a)
        self.assertEqual(bl.sig_out[0], b)

        a.state = 1.0
        bl.response()
        self.assertEqual(b.state, 2.0)

        b.sensitivity = 1.0
        self.assertWarns(Warning, bl.sensitivity)
        self.assertIsNone(a.sensitivity, msg="Default sensitivity behavior is None")

    def test_create_fail(self):
        self.assertRaises(ValueError, pym.Module.create, 'foomod1234', msg="Try to create a non-existing module")

        class FooMod(pym.Module):
            def _response(self, a_in):
                return a_in * 2

        a = pym.Signal('a')
        self.assertRaises(TypeError, FooMod, a, 1.0, msg="Try to initialize with invalid Signal object as input")

        self.assertRaises(TypeError, FooMod, [1.0, 2], a, msg="Try initializing with invalid output Signal object")

    def test_create_duplicate(self):
        class MathGeneral(pym.Module):
            def _response(self, a_in):
                return a_in * 2

        class Mathgeneral(pym.Module):
            def _response(self, a_in):
                return a_in * 2

        print(pym.Module.__subclasses__())
        self.assertWarns(Warning, pym.Module.print_children), "A warning should be emitted in case of duplicates"

        # Remove the duplicate module again
        del MathGeneral
        del Mathgeneral

        import gc
        gc.collect()

        print(pym.Module.__subclasses__())

        print("test_create_duplicate: ")
        pym.Module.print_children()

    def test_response_and_sens(self):
        class TwoInTwoOut(pym.Module):
            def _prepare(self, argument):
                self.prepared = argument

            def _response(self, a, b):
                self.internalstate = True
                return a * b, a + b

            def _sensitivity(self, dc, dd):
                self.didsensitivity = True
                a, b = [s.state for s in self.sig_in]
                return b * dc + dd, a * dc + dd

            def _reset(self):
                self.internalstate = False
                self.didsensitivity = False

        sa = pym.Signal('a', 2.5)
        sb = pym.Signal('b', 3.5)
        sc = pym.Signal('c')
        sd = pym.Signal('d')
        m = TwoInTwoOut([sa, sb], [sc, sd], 'foo')
        self.assertEqual(m.prepared, 'foo', msg="Check if the preparation has been executed")

        m.response()
        self.assertTrue(m.internalstate, msg="Check if response has been called")

        m.sensitivity()
        self.assertTrue(not hasattr(m, 'didsensitivity'), msg="Sensitivity should not have been called, "
                                                              "since output sensitivities are None")

        sc.sensitivity = 1.0
        sd.sensitivity = 1.0
        m.sensitivity()
        self.assertTrue(m.didsensitivity, msg="Sensitivity should have been called")

        self.assertEqual(sa.sensitivity, 4.5, msg="Check sensitivity value")
        self.assertEqual(sb.sensitivity, 3.5, msg="Check other sensitivity value")

        m.reset()
        self.assertFalse(m.internalstate, msg="Check if reset has worked")
        self.assertFalse(m.didsensitivity, msg="Check if reset has worked")

    def test_zero_inputs(self):
        class FooMod1(pym.Module):
            def _response(self):
                return 3.14

        b = pym.Signal('x_out')
        bl = FooMod1([], b)
        bl.response()
        self.assertEqual(b.state, 3.14)

    def test_zero_outputs(self):
        class FooMod2(pym.Module):
            def _response(self, in1):
                self.got_in1 = in1

            def _sensitivity(self):
                self.did_sens = True
                return 2.15

        a = pym.Signal('x_in', 1.256)
        bl = FooMod2(a)
        bl.response()
        self.assertEqual(bl.got_in1, 1.256, msg="State variable passed to _reponse function")

        bl.sensitivity()
        self.assertTrue(bl.did_sens, msg="Check if _sensitivity did run")
        self.assertEqual(a.sensitivity, 2.15, msg="After running first sensitivity")
        bl.sensitivity()
        self.assertEqual(a.sensitivity, 2.15 + 2.15, msg="After running second sensitivity")
        bl.reset()
        self.assertIsNone(a.sensitivity, msg="After resetting module")
        bl.sensitivity()
        self.assertEqual(a.sensitivity, 2.15, msg="First sensitivity run after reset")

    def test_wrong_response(self):
        class WrongResponse(pym.Module):
            """ Foobar

            """
            def _response(self, a):
                return a * 2.0, a * 3.0  # Two returns

        sa = pym.Signal('a', 2.5)
        sb = pym.Signal('b')
        m = WrongResponse(sa, sb)  # One output signal
        self.assertRaises(TypeError, m.response, msg="Number of out-signals should match number of returns in response")

    def test_sensitivity_and_reset_errors(self):
        class NoSensitivity(pym.Module):
            def _response(self, a, b):
                return a * b

            def _sensitivity(self, dc):
                self.did_sensitivity = True
                b = self.sig_in[1].state
                return b * dc  # Only returns one sensitivity

            def _reset(self):
                raise RuntimeError("An error has occurred")

        sa = pym.Signal('a', 2.5)
        sb = pym.Signal('b', 3.5)
        sc = pym.Signal('c')
        m = NoSensitivity([sa, sb], sc)  # Two inputs -> expects two sensitivities returned

        m.response()
        m.sensitivity()  # First test with None as sensitivity
        self.assertTrue(not hasattr(m, "did_sensitivity"))

        sc.sensitivity = 1.0
        self.assertRaises(TypeError, m.sensitivity)
        self.assertTrue(m.did_sensitivity)

        # m.reset()
        self.assertRaises(RuntimeError, m.reset)


class TestNetwork(TestCase):
    def test_correct_network(self):
        x1 = pym.Signal('x1', 2.0)
        x2 = pym.Signal('x2', 3.0)
        y1 = pym.Signal('y1')
        y2 = pym.Signal('y2')
        z = pym.Signal('z')
        m1 = pym.MathGeneral(x1, y1, expression="x1*2.0")
        m2 = pym.MathGeneral(x2, y2, expression="x2*x2 + 2.0")
        m3 = pym.MathGeneral([y1, y2], z, expression="y1*y2")

        netw1 = pym.Network(m1, m2, m3)

        netw2 = pym.Network([m1, m2, {"type": "MathGeneral", "sig_in": [y1, y2], "sig_out": z,
                                      "expression": "y1*y2"}])  # Initalize with list

        netw1.response()
        self.assertEqual(y1.state, 4.0)
        self.assertEqual(y2.state, 11.0)
        self.assertEqual(z.state, 44.0)

        netw2.response()
        self.assertEqual(y1.state, 4.0)
        self.assertEqual(y2.state, 11.0)
        self.assertEqual(z.state, 44.0)

        z.sensitivity = 1.0
        netw1.sensitivity()
        self.assertEqual(y1.sensitivity, 11.0)
        self.assertEqual(y2.sensitivity, 4.0)
        self.assertEqual(x1.sensitivity, 22.0)
        self.assertEqual(x2.sensitivity, 24.0)

        netw1.reset()
        self.assertIsNone(x1.sensitivity)
        self.assertIsNone(x2.sensitivity)
        self.assertIsNone(y1.sensitivity)
        self.assertIsNone(y2.sensitivity)
        self.assertIsNone(z.sensitivity)

    def test_network_with_initializer_error(self):
        class ErrorModule(pym.Module):
            def _response(self, a1, a2):
                raise RuntimeError("Response error")

            def _sensitivity(self, dy):
                raise KeyError("Sensitivity error")

            def _reset(self):
                raise ValueError("Reset error")

        x1 = pym.Signal('x1', 2.0)
        x2 = pym.Signal('x2', 3.0)
        y1 = pym.Signal('y1')
        y2 = pym.Signal('y2')
        z = pym.Signal('z')
        m1 = pym.MathGeneral(x1, y1, expression="x1*2.0")
        m2 = pym.MathGeneral(x2, y2, expression="x2*x2 + 2.0")
        m3 = ErrorModule([y1, y2], z)

        self.assertRaises(KeyError, pym.Network, [m1, m2, {"sig_in": [y1, y2], "sig_out": z, "expression": "y1*y2"}])

        netw = pym.Network(m1, m2, m3)
        self.assertRaises(RuntimeError, netw.response)

        z.sensitivity = 1.0
        self.assertRaises(KeyError, netw.sensitivity)
        self.assertRaises(ValueError, netw.reset)
