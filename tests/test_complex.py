import unittest
import pymoto as pym
import numpy as np
np.random.seed(0)


class ComplexVecDot(pym.Module):
    def _response(self, u, v):
        return u@v

    def _sensitivity(self, dy):
        u, v = [s.state for s in self.sig_in]
        return dy*v, dy*u


class TestComplex(unittest.TestCase):
    def test_real_to_real_via_complex(self):
        N = 4
        s_ur, s_ui = pym.Signal('ur', np.random.rand(N)), pym.Signal('ui', np.random.rand(N))
        s_vr, s_vi = pym.Signal('vr', np.random.rand(N)), pym.Signal('vi', np.random.rand(N))

        ma = pym.MakeComplex([s_ur, s_ui], pym.Signal("uc"))
        mb = pym.MakeComplex([s_vr, s_vi], pym.Signal("vc"))
        m1 = ComplexVecDot([ma.sig_out[0], mb.sig_out[0]], pym.Signal('uv'))
        m2 = pym.ComplexNorm(m1.sig_out, pym.Signal("|uv|"))

        pym.finite_difference(pym.Network(ma, mb, m1, m2), [s_ur, s_ui, s_vr, s_vi], m2.sig_out, dx=1e-5)

    def test_real_to_complex(self):
        N = 4
        s_ur, s_ui = pym.Signal('ur', np.random.rand(N)), pym.Signal('ui', np.random.rand(N) * 0)
        s_vr, s_vi = pym.Signal('vr', np.random.rand(N)), pym.Signal('vi', np.random.rand(N) * 0)

        ma = pym.MakeComplex([s_ur, s_ui], pym.Signal("uc"))
        mb = pym.MakeComplex([s_vr, s_vi], pym.Signal("vc"))
        m1 = ComplexVecDot([ma.sig_out[0], mb.sig_out[0]], pym.Signal('u.v'))

        pym.finite_difference(pym.Network(ma, mb, m1), [s_ur, s_ui, s_vr, s_vi], m1.sig_out, dx=1e-5, random=False,
                              keep_zero_structure=False)

    def test_real_to_complex1(self):
        N = 4
        s_ur, s_ui = pym.Signal('ur', np.random.rand(N)), pym.Signal('ui', np.random.rand(N))
        ma = pym.MakeComplex([s_ur, s_ui], pym.Signal("uc"))
        pym.finite_difference(ma, dx=1e-5, random=False)

    def test_complex_to_real(self):
        s_z = pym.Signal('z', np.array([0.3 + 0.4*1j]))
        m1 = pym.RealPart(s_z, pym.Signal('ur'))
        m2 = pym.ImagPart(s_z, pym.Signal('ui'))
        pym.finite_difference(pym.Network(m1, m2), s_z, [*m1.sig_out, *m2.sig_out], dx=1e-5, random=False)

    def test_complex_to_real1(self):
        s_z = pym.Signal('z', np.array([0.3 + 0.4*1j]))
        m = pym.ComplexNorm(s_z, pym.Signal('|z|'))
        pym.finite_difference(m, dx=1e-5, random=False)

    def test_complex_to_real2(self):
        N = 4
        u = np.random.rand(N) + 1j * np.random.rand(N)
        v = np.random.rand(N) + 1j * np.random.rand(N)
        m1 = ComplexVecDot([pym.Signal('u', u), pym.Signal('v', v)], pym.Signal('u'))
        m2 = pym.ComplexNorm(m1.sig_out, pym.Signal("ur"))
        pym.finite_difference(pym.Network(m1, m2), m1.sig_in, m2.sig_out, dx=1e-5)

    def test_complex_to_complex(self):
        N = 4
        u = np.random.rand(N) + 1j * np.random.rand(N)
        v = np.random.rand(N) + 1j * np.random.rand(N)
        m1 = ComplexVecDot([pym.Signal('u', u), pym.Signal('v', v)], pym.Signal('u.v'))
        pym.finite_difference(m1, dx=1e-5)
