from unittest import TestCase
import pyModular as pym
import numpy as np


class TestSignal(TestCase):
    def test_initialize(self):
        a = pym.Signal('foo')
        self.assertEqual(a.tag, 'foo', msg="Initialize tag")
        self.assertIsNone(a.get_state(), msg="Initialize state to None")
        self.assertIsNone(a.get_sens(), msg="Initialize sensitivity to None")

    def test_state(self):
        a = pym.Signal('foo')
        a.set_state(1.0)
        self.assertEqual(a.get_state(), 1.0, msg="Set state to scalar")

        a.set_state(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(a.get_state(), np.array([1.0, 2.0, 3.0])), msg="Set state to array")

    def test_sensitivity(self):
        a = pym.Signal('foo')
        a.set_sens(2.0)
        self.assertEqual(a.get_sens(), 2.0, msg="Set initial sensitivity to scalar")

        a.add_sens(3.0)
        self.assertEqual(a.get_sens(), 5.0, msg="Add scalar sensitivity")

        a.set_sens(1.0)
        self.assertEqual(a.get_sens(), 1.0, msg="Rewrite sensitivity by set_sens")

        a.reset()
        self.assertEqual(a.get_sens(), None, msg="Reset sensitivity")

        a.add_sens(np.array([1.1, 2.2, 3.3]))
        self.assertTrue(np.allclose(a.get_sens(), np.array([1.1, 2.2, 3.3])), msg="Set initial sensitivity by add_sens")

        a.add_sens(None)
        self.assertTrue(np.allclose(a.get_sens(), np.array([1.1, 2.2, 3.3])), msg="After adding None by add_sens")

        self.assertRaises(ValueError, a.add_sens, np.random.rand(3, 3))

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

        bl = pym.Module(a, b)

        self.assertEqual(len(bl.sig_in), 1)
        self.assertEqual(len(bl.sig_out), 1)
        self.assertEqual(bl.sig_in[0], a)
        self.assertEqual(bl.sig_out[0], b)

        # Response with unset input state should return error
        self.assertRaises(NotImplementedError, bl.response)

    def test_initialize2(self):
        a = pym.Signal('x1')
        b = pym.Signal('x2')
        c = pym.Signal('x3')
        d = pym.Signal('y1')
        e = pym.Signal('y2')

        bl = pym.Module([a, b, c], [d, e])

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

        a.set_state(1.0)
        bl.response()
        self.assertEqual(b.get_state(), 2.0)

        b.set_sens(1.0)
        self.assertWarns(Warning, bl.sensitivity)
        self.assertIsNone(a.get_sens(), msg="Default sensitivity behavior is None")

    def test_create_fail(self):
        self.assertRaises(ValueError, pym.Module.create, 'foomod1234', msg="Try to create a non-existing module")

        a = pym.Signal('a')
        self.assertRaises(TypeError, pym.Module, a, 1.0, msg="Try to initialize with invalid Signal object as input")

        self.assertRaises(TypeError, pym.Module, [1.0, 2], a, msg="Try initializing with invalid output Signal object")

    def test_create_duplicate(self):
        class Mathgeneral(pym.Module):
            def _response(self, a_in):
                return a_in * 2

        class mathgeneral(pym.Module):
            def _response(self, a_in):
                return a_in * 2

        print(pym.Module.__subclasses__())
        print("test_create_duplicate assertion: ")
        self.assertWarns(Warning, pym.Module.print_children)

        # Remove the duplicate module again
        del Mathgeneral
        del mathgeneral

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
                a = self.sig_in[0].get_state()
                b = self.sig_in[1].get_state()
                return b * dc + dd, a * dc + dd

            def _reset(self):
                self.internalstate = False
                self.didsensitivity = False

        sa = pym.Signal('a')
        sa.set_state(2.5)
        sb = pym.Signal('b')
        sb.set_state(3.5)
        sc = pym.Signal('c')
        sd = pym.Signal('d')
        m = TwoInTwoOut([sa, sb], [sc, sd], 'arg')
        self.assertEqual(m.prepared, 'arg')

        m.response()
        self.assertTrue(m.internalstate)

        m.sensitivity()
        self.assertTrue(not hasattr(m, 'didsensitivity'))

        sc.set_sens(1.0)
        sd.set_sens(1.0)
        m.sensitivity()
        self.assertTrue(m.didsensitivity)

        self.assertEqual(sa.get_sens(), 4.5)
        self.assertEqual(sb.get_sens(), 3.5)

        m.reset()
        self.assertFalse(m.internalstate)
        self.assertFalse(m.didsensitivity)

    def test_zero_inputs(self):
        class FooMod1(pym.Module):
            def _response(self):
                return 3.14

        print("test_zero_inputs: ")
        pym.Module.print_children()
        b = pym.Signal('x_out')
        bl = FooMod1([], b)
        bl.response()
        self.assertEqual(b.get_state(), 3.14)

    def test_zero_outputs(self):
        class FooMod2(pym.Module):
            def _response(self, in1):
                self.got_in1 = in1

            def _sensitivity(self):
                self.did_sens = True
                return 2.15

        a = pym.Signal('x_in')
        a.set_state(1.256)
        bl = FooMod2(a)
        bl.response()
        self.assertEqual(bl.got_in1, 1.256, msg="State variable passed to _reponse function")

        bl.sensitivity()
        self.assertTrue(bl.did_sens, msg="Check if _sensitivity did run")
        self.assertEqual(a.get_sens(), 2.15, msg="After running first sensitivity")
        bl.sensitivity()
        self.assertEqual(a.get_sens(), 2.15 + 2.15, msg="After running second sensitivity")
        bl.reset()
        self.assertIsNone(a.get_sens(), msg="After resetting module")
        bl.sensitivity()
        self.assertEqual(a.get_sens(), 2.15, msg="First sensitivity run after reset")

    def test_wrong_response(self):
        class WrongResponse(pym.Module):
            def _response(self, a):
                return a * 2.0, a * 3.0  # Two returns

        sa = pym.Signal('a')
        sa.set_state(2.5)
        sb = pym.Signal('b')
        m = WrongResponse(sa, sb)  # One output signal

        self.assertRaises(TypeError, m.response, msg="Number of out-signals should match number of returns in response")

    def test_sensitivity_and_reset_errors(self):
        class NoSensitivity(pym.Module):
            def _response(self, a, b):
                return a * b

            def _sensitivity(self, dc):
                self.did_sensitivity = True
                b = self.sig_in[1].get_state()
                return b * dc  # Only returns one sensitivity

            def _reset(self):
                raise RuntimeError("An error has occurred")

        sa = pym.Signal('a')
        sa.set_state(2.5)
        sb = pym.Signal('b')
        sb.set_state(3.5)
        sc = pym.Signal('c')
        m = NoSensitivity([sa, sb], sc)  # Two inputs -> expects two sensitivities returned

        m.response()
        m.sensitivity()  # First test with None as sensitivity
        self.assertTrue(not hasattr(m, "did_sensitivity"))

        sc.set_sens(1.0)
        self.assertRaises(TypeError, m.sensitivity)
        self.assertTrue(m.did_sensitivity)

        self.assertRaises(RuntimeError, m.reset)


class TestNetwork(TestCase):
    def test_correct_network(self):
        x1 = pym.Signal('x1')
        x2 = pym.Signal('x2')
        y1 = pym.Signal('y1')
        y2 = pym.Signal('y2')
        z = pym.Signal('z')
        m1 = pym.MathGeneral(x1, y1, expression="x1*2.0")
        m2 = pym.MathGeneral(x2, y2, expression="x2*x2 + 2.0")
        m3 = pym.MathGeneral([y1, y2], z, expression="y1*y2")

        netw1 = pym.Network(m1, m2, m3)

        netw2 = pym.Network([m1, m2, {"type": "MathGeneral", "sig_in": [y1, y2], "sig_out": z,
                                      "expression": "y1*y2"}])  # Initalize with list

        x1.set_state(2.0)
        x2.set_state(3.0)
        netw1.response()
        self.assertEqual(y1.get_state(), 4.0)
        self.assertEqual(y2.get_state(), 11.0)
        self.assertEqual(z.get_state(), 44.0)

        netw2.response()
        self.assertEqual(y1.get_state(), 4.0)
        self.assertEqual(y2.get_state(), 11.0)
        self.assertEqual(z.get_state(), 44.0)

        z.set_sens(1.0)
        netw1.sensitivity()
        self.assertEqual(y1.get_sens(), 11.0)
        self.assertEqual(y2.get_sens(), 4.0)
        self.assertEqual(x1.get_sens(), 22.0)
        self.assertEqual(x2.get_sens(), 24.0)

        netw1.reset()
        self.assertIsNone(x1.get_sens())
        self.assertIsNone(x2.get_sens())
        self.assertIsNone(y1.get_sens())
        self.assertIsNone(y2.get_sens())
        self.assertIsNone(z.get_sens())

    def test_network_with_initializer_error(self):
        class ErrorModule(pym.Module):
            def _response(self, a1, a2):
                raise RuntimeError("Response error")

            def _sensitivity(self, dy):
                raise KeyError("Sensitivity error")

            def _reset(self):
                raise ValueError("Reset error")

        x1 = pym.Signal('x1')
        x2 = pym.Signal('x2')
        y1 = pym.Signal('y1')
        y2 = pym.Signal('y2')
        z = pym.Signal('z')
        m1 = pym.MathGeneral(x1, y1, expression="x1*2.0")
        m2 = pym.MathGeneral(x2, y2, expression="x2*x2 + 2.0")
        m3 = ErrorModule([y1, y2], z)

        self.assertRaises(KeyError, pym.Network, [m1, m2, {"sig_in": [y1, y2], "sig_out": z, "expression": "y1*y2"}])

        netw = pym.Network(m1, m2, m3)
        x1.set_state(2.0)
        x2.set_state(3.0)
        self.assertRaises(RuntimeError, netw.response)

        z.set_sens(1.0)
        self.assertRaises(KeyError, netw.sensitivity)
        self.assertRaises(ValueError, netw.reset)
