import pytest
import numpy.testing as npt
import pymoto as pym
import numpy as np


class TestNetwork:
    def test_network_wsignals(self):
        x1 = pym.Signal('x1', 2.0)
        x2 = pym.Signal('x2', 3.0)

        with pym.Network() as netw:
            y1 = pym.MathGeneral("x1*2.0")(x1)
            y1.tag = 'y1'
            y2 = pym.MathGeneral("x2*x2 + 2.0")(x2)
            y2.tag = 'y2'
            z = pym.MathGeneral("y1*y2")(y1, y2)

        netw.response()
        assert y1.state == 4.0
        assert y2.state == 11.0
        assert z.state == 44.0

        z.sensitivity = 1.0
        netw.sensitivity()
        assert y1.sensitivity == 11.0
        assert y2.sensitivity == 4.0
        assert x1.sensitivity == 22.0
        assert x2.sensitivity == 24.0

        netw.reset()
        assert x1.sensitivity is None
        assert x2.sensitivity is None
        assert y1.sensitivity is None
        assert y2.sensitivity is None
        assert z.sensitivity is None

    def test_reconnect_module_add_twice_in_network(self):
        class MyMod(pym.Module):
            def __call__(self, A, B):
                return A + B

            def _sensitivity(self, dC):
                return dC, dC

        s_A = pym.Signal('A', np.array([1, 2, 3]))
        s_B = pym.Signal('B', np.array([4, 5, 6]))
        s_C = pym.Signal('C', np.array([1, 2]))
        s_D = pym.Signal('D', np.array([4, 5]))
        m = MyMod()
        with pym.Network() as fn:
            s_E = m(s_A, s_B)
            pytest.raises(RuntimeError, m, s_C, s_D)

    def test_reconnect_module_add_twice_in_network_copy(self):
        class MyMod(pym.Module):
            def __call__(self, A, B):
                return A + B

            def _sensitivity(self, dC):
                return dC, dC

        s_A = pym.Signal('A', np.array([1, 2, 3]))
        s_B = pym.Signal('B', np.array([4, 5, 6]))
        s_C = pym.Signal('C', np.array([1, 2]))
        s_D = pym.Signal('D', np.array([4, 5]))
        m = MyMod()
        import copy
        with pym.Network() as fn:
            s_E = m(s_A, s_B)
            s_F = copy.deepcopy(m)(s_C, s_D)  # Use copy to create a new module
        npt.assert_equal(s_E.state, np.array([5, 7, 9]))
        npt.assert_equal(s_F.state, np.array([5, 7]))

class TestSubsets:
    class SourceMod(pym.Module):
        def __call__(self):
            return 1.0

    class SinkMod(pym.Module):
        def __call__(self, x):
            pass

    class Func1to1Mod(pym.Module):
        def __call__(self, x):
            return x*2

    class Func1to2Mod(pym.Module):
        def __call__(self, x):
            return x*2, x*3

    class Func2to1Mod(pym.Module):
            def __call__(self, x, y):
                return x + y

    def test_input_output_cone(self):
        sx = pym.Signal('x', 1.0)
        sy = pym.Signal('y', 2.0)

        with pym.Network() as fn:
            m1 = self.Func1to2Mod()
            m2 = self.Func1to1Mod()
            m2a = self.Func1to1Mod()
            m3 = self.SinkMod()
            m4 = self.SourceMod()
            m5 = self.Func1to1Mod()
            m6 = self.Func2to1Mod()
            m7 = self.Func2to1Mod()
            m8 = self.Func2to1Mod()
            m9 = self.Func1to1Mod()
            m10 = self.Func2to1Mod()

            sx1, sx2 = m1(sx)  # sx -> sx1, sx2
            sy1 = m2(sy)  # sy -> sy1, sy1a

            # sink branch
            sy1a = m2a(sy1)  # sy, sy1 -> sy1a
            m3(sy1a)  # sy, sy1, sy1a -> ..
            # ---
            sz = m4()  # .. -> sz, sz1, so1
            sz1 = m5(sz)  # sz -> sz1, so1
            so1 = m6(sz1, sx2)  # sz, sz1, sx, sx2 -> so1
            so2 = m7(sy1, sx1)  # sy, sy1, sx, sx1
            sg1 = m8(so1, so2)  # sz, sz1, sx, sx1, sx2, so1, so2
            sg2 = m9(sy1)  # sy, sy1
            sg3 = m10(sg1, 2.5)  # sy, sy1, sz, sz1, sx, sx1, sx2, so1, so2, sg1

            sx1.tag = 'x1'
            sx2.tag = 'x2'
            sy1.tag = 'y1'
            sy1a.tag = 'y1a'
            sz.tag = 'z'
            sz1.tag = 'z1'
            so1.tag = 'o1'
            so2.tag = 'o2'
            sg1.tag = 'g1'
            sg2.tag = 'g2'
            sg3.tag = 'g3'

            # Disconnected part
            dm1 = self.SourceMod()
            dm2 = self.Func1to1Mod()
            dm3 = self.SinkMod()
            sd1 = dm1()
            sd1.tag = 'd1'
            sd2 = dm2(sd1)
            sd2.tag = 'd2'
            dm3(sd2)

        all_mods = set(fn.mods)

        testcases = [
            dict(fn=fn.get_input_cone(sd1), incl={dm2, dm3}),
            dict(fn=fn.get_input_cone(so1), incl={m8, m10}),
            dict(fn=fn.get_input_cone(frommod=dm1), incl={dm1, dm2, dm3}),
            dict(fn=fn.get_input_cone([sy1, so1]), incl={m2a, m3, m7, m8, m9, m10}),
            dict(fn=fn.get_input_cone(frommod=set()), incl=all_mods),
            dict(fn=fn.get_input_cone(frommod=None), incl=all_mods),
            dict(fn=fn.get_input_cone(fromsig=set()), incl=all_mods),
            dict(fn=fn.get_input_cone(fromsig=None), incl=all_mods),
            dict(fn=fn.get_output_cone(sd2), incl={dm2, dm1}),
            dict(fn=fn.get_output_cone(so1), incl={m6, m5, m4, m1}),
            dict(fn=fn.get_output_cone(tomod=dm3), incl={dm1, dm2, dm3}),
            dict(fn=fn.get_output_cone([so2, so1]), incl={m7, m6, m5, m4, m2, m1}),
            dict(fn=fn.get_output_cone(tomod=set()), incl=all_mods),
            dict(fn=fn.get_output_cone(tomod=None), incl=all_mods),
            dict(fn=fn.get_output_cone(tosig=set()), incl=all_mods),
            dict(fn=fn.get_output_cone(tosig=None), incl=all_mods),
            dict(fn=fn.get_input_cone(fromsig=sd1, frommod=dm1), incl={dm1, dm2, dm3}),
            dict(fn=fn.get_output_cone(tosig=sd2, tomod=dm3), incl={dm1, dm2, dm3}),
            dict(fn=fn.get_input_cone(fromsig=sd1, frommod=dm1).get_output_cone(tosig=sd2, tomod=dm3), incl={dm1, dm2, dm3}),
            dict(fn=fn.get_output_cone(tosig=sd2, tomod=dm3).get_input_cone(fromsig=sd1, frommod=dm1), incl={dm1, dm2, dm3}),
        ]

        for t in testcases:
            incl = t['incl'].copy()
            expected_mods = ''.join([f'\n\t- {s}' for s in t['incl']])
            errstr = f"Function ({len(t['fn'])}) = {t['fn']}\nExpected ({len(t['incl'])}) = {expected_mods}"
            for m in t['fn']:
                assert m in incl, errstr
                incl.remove(m)
            assert len(incl) == 0, errstr

    @pytest.mark.parametrize('include_sinks', [True, False])
    @pytest.mark.parametrize('include_sources', [True, False])
    def test_subset(self, include_sinks, include_sources):
        opts = dict(
            include_sinks=include_sinks,
            include_sources=include_sources,
        )
        sx = pym.Signal('x', 1.0)
        sy = pym.Signal('y', 2.0)

        with pym.Network() as fn:
            m1 = self.Func1to2Mod()
            m2 = self.Func1to1Mod()
            m2a = self.Func1to1Mod()
            m3 = self.SinkMod()
            m4 = self.SourceMod()
            m5 = self.Func1to1Mod()
            m6 = self.Func2to1Mod()
            m7 = self.Func2to1Mod()
            m8 = self.Func2to1Mod()
            m9 = self.Func1to1Mod()
            m10 = self.Func2to1Mod()

            sx1, sx2 = m1(sx)  # sx -> sx1, sx2
            sy1 = m2(sy)  # sy -> sy1, sy1a

            #sink branch
            sy1a = m2a(sy1)  # sy, sy1 -> sy1a
            m3(sy1a)  # sy, sy1, sy1a -> ..
            # ---
            sz = m4()  # .. -> sz, sz1, so1
            sz1 = m5(sz)  # sz -> sz1, so1
            so1 = m6(sz1, sx2)  # sz, sz1, sx, sx2 -> so1
            so2 = m7(sy1, sx1)  # sy, sy1, sx, sx1
            sg1 = m8(so1, so2)  # sz, sz1, sx, sx1, sx2, so1, so2
            sg2 = m9(sy1)  # sy, sy1
            sg3 = m10(sg1, 2.5)  # sy, sy1, sz, sz1, sx, sx1, sx2, so1, so2, sg1


            sx1.tag = 'x1'
            sx2.tag = 'x2'
            sy1.tag = 'y1'
            sy1a.tag = 'y1a'
            sz.tag = 'z'
            sz1.tag = 'z1'
            so1.tag = 'o1'
            so2.tag = 'o2'
            sg1.tag = 'g1'
            sg2.tag = 'g2'
            sg3.tag = 'g3'

            # Disconnected part
            dm1 = self.SourceMod()
            dm2 = self.Func1to1Mod()
            dm3 = self.SinkMod()
            sd1 = dm1()
            sd1.tag = 'd1'
            sd2 = dm2(sd1)
            sd2.tag = 'd2'
            dm3(sd2)

        all_mods = set(fn.mods)
        sink_mods = {m2a, m3} if include_sinks else {}  # sy, sy1, sy1a -> sink
        source_mods = {m4, m5} if include_sources else {}

        testcases = [
            dict(fn=fn.get_subset(fromsig=sx, **opts), incl={m1, m6, m7, m8, m10}.union(source_mods)),
            dict(fn=fn.get_subset(fromsig=[sx, sy], **opts), incl={m1, m2, m6, m7, m8, m9, m10}.union(source_mods).union(sink_mods)),
            dict(fn=fn.get_subset(fromsig=sx2, **opts), incl={m6, m8, m10}.union(source_mods)),
            dict(fn=fn.get_subset(tosig=sg1, **opts), incl={m1, m2, m6, m7, m8}.union(source_mods).union(sink_mods)),
            dict(fn=fn.get_subset(tosig=[sg2, sx2], **opts), incl={m1, m2, m9}.union(sink_mods)),
            dict(fn=fn.get_subset(tosig=sg3, **opts), incl={m1, m2, m6, m7, m8, m10}.union(source_mods).union(sink_mods)),
            dict(fn=fn.get_subset(fromsig=sd1, tosig=sd2, **opts), incl={dm2}.union({dm1} if include_sources else {}).union({dm3} if include_sinks else {})),
        ]

        for t in testcases:
            incl = t['incl'].copy()
            expected_mods = ''.join([f'\n\t- {s}' for s in t['incl']])
            errstr = f"Function ({len(t['fn'])}) = {t['fn']}\nExpected ({len(t['incl'])}) = {expected_mods}"
            for m in t['fn']:
                assert m in incl, errstr
                incl.remove(m)
            assert len(incl) == 0, errstr

    def test_slice(self):
        sx = pym.Signal('x', 1.0)
        sy = pym.Signal('y', 2.0)

        with pym.Network() as fn:
            m1 = self.Func1to2Mod()
            m2 = self.Func1to1Mod()
            m2a = self.Func1to1Mod()
            m3 = self.SinkMod()
            m4 = self.SourceMod()
            m5 = self.Func1to1Mod()
            m6 = self.Func2to1Mod()
            m7 = self.Func2to1Mod()
            m8 = self.Func2to1Mod()
            m9 = self.Func1to1Mod()
            m10 = self.Func2to1Mod()

            sx1, sx2 = m1(sx)  # sx -> sx1, sx2
            sy1 = m2(sy)  # sy -> sy1, sy1a

            #sink branch
            sy1a = m2a(sy1)  # sy, sy1 -> sy1a
            m3(sy1a)  # sy, sy1, sy1a -> ..
            # ---
            sz = m4()  # .. -> sz, sz1, so1
            sz1 = m5(sz)  # sz -> sz1, so1
            so1 = m6(sz1, sx2)  # sz, sz1, sx, sx2 -> so1
            so2 = m7(sy1, sx1)  # sy, sy1, sx, sx1
            sg1 = m8(so1, so2)  # sz, sz1, sx, sx1, sx2, so1, so2
            sg2 = m9(sy1)  # sy, sy1
            sg3 = m10(sg1, 2.5)  # sy, sy1, sz, sz1, sx, sx1, sx2, so1, so2, sg1


            sx1.tag = 'x1'
            sx2.tag = 'x2'
            sy1.tag = 'y1'
            sy1a.tag = 'y1a'
            sz.tag = 'z'
            sz1.tag = 'z1'
            so1.tag = 'o1'
            so2.tag = 'o2'
            sg1.tag = 'g1'
            sg2.tag = 'g2'
            sg3.tag = 'g3'

            # Disconnected part
            dm1 = self.SourceMod()
            dm2 = self.Func1to1Mod()
            dm3 = self.SinkMod()
            sd1 = dm1()
            sd1.tag = 'd1'
            sd2 = dm2(sd1)
            sd2.tag = 'd2'
            dm3(sd2)

        all_mods = set(fn.mods)

        # Slice single module
        assert fn[0] == m1
        assert fn[-1] == dm3
        assert fn[sd1:sd2] == dm2

        testcases = [
            dict(fn=fn[1:4], incl={m2, m2a, m3}),  # Linear indexing in list
            dict(fn=fn[:4], incl={m1, m2, m2a, m3}),
            dict(fn=fn[-3:], incl={dm1, dm2, dm3}),
            dict(fn=fn[sd1:], incl={dm2, dm3}),  # Connectivity-based indexing
            dict(fn=fn[:sd2], incl={dm1, dm2}),
            dict(fn=fn[:], incl=all_mods)
        ]

        for t in testcases:
            incl = t['incl'].copy()
            expected_mods = ''.join([f'\n\t- {s}' for s in t['incl']])
            errstr = f"Function ({len(t['fn'])}) = {t['fn']}\nExpected ({len(t['incl'])}) = {expected_mods}"
            for m in t['fn']:
                assert m in incl, errstr
                incl.remove(m)
            assert len(incl) == 0, errstr

    def test_with_sliced_signals(self):
        sx = pym.Signal('x', np.array([1, 2, 3]))

        with pym.Network() as fn:
            dm1 = self.SourceMod()
            dm2 = self.Func2to1Mod()
            dm3 = self.SinkMod()
            sd1 = dm1()
            sd1.tag = 'd1'
            sd2 = dm2(sd1, sx[2])
            sd2.tag = 'd2'
            dm3(sd2)

        testcases = [
            dict(fn=fn.get_input_cone(sx), incl={dm2, dm3}),
            dict(fn=fn.get_output_cone(sd2[0]), incl={dm1, dm2}),
            dict(fn=fn.get_subset(sx[1], sd2[0], include_sinks=False, include_sources=False), incl={dm2}),
            dict(fn=fn.get_subset(sx[1], sd2[0], include_sinks=True, include_sources=True), incl={dm1, dm2, dm3}),
        ]

        for t in testcases:
            incl = t['incl'].copy()
            expected_mods = ''.join([f'\n\t- {s}' for s in t['incl']])
            errstr = f"Function ({len(t['fn'])}) = {t['fn']}\nExpected ({len(t['incl'])}) = {expected_mods}"
            for m in t['fn']:
                assert m in incl, errstr
                incl.remove(m)
            assert len(incl) == 0, errstr

