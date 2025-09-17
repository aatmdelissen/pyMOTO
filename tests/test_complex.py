import pytest
import pymoto as pym
import numpy as np
import numpy.testing as npt


class ComplexVecDot(pym.Module):
    def __call__(self, u, v):
        return u@v

    def _sensitivity(self, dy):
        u, v = self.get_input_states()
        return dy*v, dy*u


class TestComplex:
    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

    def test_real_to_real_via_complex(self):
        np.random.seed(0)
        N = 4
        s_ur, s_ui = pym.Signal('ur', np.random.rand(N)), pym.Signal('ui', np.random.rand(N))
        s_vr, s_vi = pym.Signal('vr', np.random.rand(N)), pym.Signal('vi', np.random.rand(N))

        s_uc = pym.MakeComplex()(s_ur, s_ui)
        s_uc.tag = 'uc'
        s_vc = pym.MakeComplex()(s_vr, s_vi)
        s_vc.tag = 'vc'
        s_uv = ComplexVecDot()(s_uc, s_vc)
        s_uv.tag = 'u.v'
        s_uvn = pym.ComplexNorm()(s_uv)
        s_uvn.tag = '|u.v|'

        assert s_uvn.state == np.abs((s_ur.state + 1j*s_ui.state) @ (s_vr.state + 1j*s_vi.state))

        pym.finite_difference([s_ur, s_ui, s_vr, s_vi], s_uvn, dx=1e-5, test_fn=self.fd_testfn)

    def test_real_to_complex(self):
        np.random.seed(0)
        N = 4
        s_ur, s_ui = pym.Signal('ur', np.random.rand(N)), pym.Signal('ui', np.random.rand(N) * 0)
        s_vr, s_vi = pym.Signal('vr', np.random.rand(N)), pym.Signal('vi', np.random.rand(N) * 0)

        s_uc = pym.MakeComplex()(s_ur, s_ui)
        s_uc.tag = 'uc'
        s_vc = pym.MakeComplex()(s_vr, s_vi)
        s_vc.tag = 'vc'
        s_uv = ComplexVecDot()(s_uc, s_vc)
        s_uv.tag = 'u.v'

        pym.finite_difference([s_ur, s_ui, s_vr, s_vi], s_uv, dx=1e-5, random=False, 
                              keep_zero_structure=False, test_fn=self.fd_testfn)

    def test_real_to_complex1(self):
        np.random.seed(0)
        N = 4
        s_ur, s_ui = pym.Signal('ur', np.random.rand(N)), pym.Signal('ui', np.random.rand(N))
        s_uc = pym.MakeComplex()(s_ur, s_ui)
        s_uc.tag = 'uc'
        pym.finite_difference([s_ur, s_ui], s_uc, dx=1e-5, random=False, test_fn=self.fd_testfn)

    def test_complex_to_real(self):
        s_z = pym.Signal('z', np.array([0.3 + 0.4*1j]))
        s_zr = pym.RealPart()(s_z)
        s_zi = pym.ImagPart()(s_z)
        assert s_zr.state == 0.3
        assert s_zi.state == 0.4
        pym.finite_difference(s_z, [s_zr, s_zi], dx=1e-5, random=False, test_fn=self.fd_testfn)

    def test_complex_to_real_split(self):
        s_z = pym.Signal('z', np.array([0.3 + 0.4 * 1j]))
        s_zr, s_zi = pym.SplitComplex()(s_z)
        assert s_zr.state == 0.3
        assert s_zi.state == 0.4
        pym.finite_difference(s_z, [s_zr, s_zi], dx=1e-5, random=False, test_fn=self.fd_testfn)

    def test_complex_to_real1(self):
        s_z = pym.Signal('z', np.array([0.3 + 0.4*1j]))
        s_zn = pym.ComplexNorm()(s_z)
        assert s_zn.state == np.sqrt(0.3**2 + 0.4**2)
        pym.finite_difference(s_z, s_zn, dx=1e-5, random=False, test_fn=self.fd_testfn)

    def test_complex_to_real2(self):
        np.random.seed(0)
        N = 4
        s_u = pym.Signal('u', state=np.random.rand(N) + 1j * np.random.rand(N))
        s_v = pym.Signal('v', state=np.random.rand(N) + 1j * np.random.rand(N))

        s_uv = ComplexVecDot()(s_u, s_v)
        s_uvn = pym.ComplexNorm()(s_uv)
        pym.finite_difference([s_u, s_v], [s_uv, s_uvn], dx=1e-5, test_fn=self.fd_testfn)

    def test_complex_to_complex(self):
        np.random.seed(0)
        N = 4
        s_u = pym.Signal('u', state=np.random.rand(N) + 1j * np.random.rand(N))
        s_v = pym.Signal('v', state=np.random.rand(N) + 1j * np.random.rand(N))

        s_uv = ComplexVecDot()(s_u, s_v)
        pym.finite_difference([s_u, s_v], s_uv, dx=1e-5, test_fn=self.fd_testfn)

    def test_conjugate(self):
        np.random.seed(0)
        N = 4
        s_u = pym.Signal('u', state=np.random.rand(N) + 1j * np.random.rand(N))
        s_v = pym.Conjugate()(s_u)
        npt.assert_allclose(s_v.state, np.conj(s_u.state))
        pym.finite_difference(s_u, s_v, test_fn=self.fd_testfn)


if __name__ == '__main__':
    pytest.main()