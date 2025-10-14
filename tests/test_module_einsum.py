import pytest
import pymoto as pym
import numpy as np
import numpy.testing as npt


def make_vec(n):
    return np.random.rand(n)


def make_mat(m, n=None):
    if n is None:
        n = m
    return np.random.rand(m, n)


class TestEinSum:
    np.random.seed(0)

    @staticmethod
    def fd_testfn(x0, dx, df_an, df_fd):
        npt.assert_allclose(df_an, df_fd, rtol=1e-5, atol=1e-15)

    @pytest.mark.parametrize('a', [make_vec(4), 1j*make_vec(5), make_vec(6)+1j*make_vec(6)], 
                             ids=['real', 'imaginary', 'complex'])
    def test_vec_sum(self, a):
        out_chk = np.sum(a)

        s_a = pym.Signal("vec", a)
        s_out = pym.EinSum("i->")(s_a)
        s_out.tag = 'sum(a)'

        npt.assert_allclose(out_chk, s_out.state)
        pym.finite_difference(tosig=s_out, test_fn=self.fd_testfn)

    @pytest.mark.parametrize('b', [make_vec(4), 1j*make_vec(4), make_vec(4)+1j*make_vec(4)], 
                             ids=['real', 'imaginary', 'complex'])
    @pytest.mark.parametrize('a', [make_vec(4), 1j*make_vec(4), make_vec(4)+1j*make_vec(4)], 
                             ids=['real', 'imaginary', 'complex'])
    def test_vec_dot(self, a, b):
        out_chk = a@b

        s_a = pym.Signal("veca", a.copy())
        s_b = pym.Signal("vecb", b.copy())

        s_out = pym.EinSum("i,i->")(s_a, s_b)
        s_out.tag = 'a.b'

        npt.assert_allclose(out_chk, s_out.state)
        pym.finite_difference(tosig=s_out, test_fn=self.fd_testfn)

    @pytest.mark.parametrize('b', [make_vec(5), 1j*make_vec(5), make_vec(5)+1j*make_vec(5)], 
                             ids=['real', 'imaginary', 'complex'])
    @pytest.mark.parametrize('a', [make_vec(5), 1j*make_vec(5), make_vec(5)+1j*make_vec(5)], 
                             ids=['real', 'imaginary', 'complex'])
    def test_vec_outer(self, a, b):
        out_chk = np.outer(a, b)
        a_tag = "r" if np.isrealobj(a) else ("c" if np.linalg.norm(np.real(a)) != 0 else "i")
        b_tag = "r" if np.isrealobj(b) else ("c" if np.linalg.norm(np.real(b)) != 0 else "i")

        s_a = pym.Signal(f"a_{a_tag}", a.copy())
        s_b = pym.Signal(f"b_{b_tag}", b.copy())

        s_out = pym.EinSum("i,j->ij")(s_a, s_b)
        s_out.tag = f'outer({s_a.tag}, {s_b.tag})'

        npt.assert_allclose(out_chk, s_out.state)

        pym.finite_difference(tosig=s_out, test_fn=self.fd_testfn)

    @pytest.mark.parametrize('a', [make_mat(4), 1j*make_mat(5), make_mat(6) + 1j * make_mat(6)], 
                             ids=['real', 'imaginary', 'complex'])
    def test_mat_trace(self, a):
        out_chk = np.trace(a)

        s_a = pym.Signal("mat", a.copy())

        s_out = pym.EinSum("ii->")(s_a)
        s_out.tag = "trace(A)"

        npt.assert_allclose(out_chk, s_out.state)
        pym.finite_difference(tosig=s_out, test_fn=self.fd_testfn)

    @pytest.mark.parametrize('a', [make_mat(4), 1j*make_mat(5), make_mat(6) + 1j * make_mat(6)], 
                             ids=['real', 'imaginary', 'complex'])
    def test_mat_sum(self, a):
        out_chk = np.sum(a)

        s_a = pym.Signal("mat", a.copy())

        s_out = pym.EinSum("ij->")(s_a)
        s_out.tag = 'sum(A)'

        npt.assert_allclose(out_chk, s_out.state)
        pym.finite_difference(tosig=s_out, test_fn=self.fd_testfn)

    def test_einsum_mat_diag(self):
        n = 3
    
        a = np.random.rand(n, n)
    
        out_chk = np.diag(a)
    
        s_a = pym.Signal("mat", a)

        s_out = pym.EinSum("ii->i")(s_a)
        s_out.tag = "diag(A)"
        
        npt.assert_allclose(out_chk, s_out.state)

        # TODO: Sensitivity does not work for this, as the sensitivity would have repeated index in the output
        pytest.raises(NotImplementedError, pym.finite_difference, tosig=s_out, test_fn=self.fd_testfn)
        # pym.finite_difference(test_fn=self.fd_testfn)

    @pytest.mark.parametrize('B', [make_mat(4), 1j*make_mat(4), make_mat(4) + 1j * make_mat(4)], 
                             ids=['real', 'imaginary', 'complex'])
    @pytest.mark.parametrize('A', [make_mat(4), 1j*make_mat(4), make_mat(4) + 1j * make_mat(4)], 
                             ids=['real', 'imaginary', 'complex'])
    def test_matmat(self, A, B):
        out_chk = A.dot(B)

        s_a = pym.Signal("matA", A.copy())
        s_b = pym.Signal("matB", B.copy())

        s_out = pym.EinSum("ij,jk->ik")(s_a, s_b)
        s_out.tag = 'A.B'

        npt.assert_allclose(out_chk, s_out.state)
        pym.finite_difference(tosig=s_out, test_fn=self.fd_testfn)

    @pytest.mark.parametrize('v', [make_vec(4), 1j * make_vec(4), make_vec(4) + 1j * make_vec(4)], 
                             ids=['real', 'imaginary', 'complex'])
    @pytest.mark.parametrize('B', [make_mat(4), 1j * make_mat(4), make_mat(4) + 1j * make_mat(4)], 
                             ids=['real', 'imaginary', 'complex'])
    @pytest.mark.parametrize('A', [make_mat(3, 4), 1j * make_mat(2, 4), make_mat(3, 4) + 1j * make_mat(3, 4)], 
                             ids=['real', 'imaginary', 'complex'])
    def test_matmatvec(self, A, B, v):
        out_chk = np.einsum("ij,jk,k->ik", A, B, v)

        s_A = pym.Signal("A", A.copy())
        s_B = pym.Signal("B", B.copy())
        s_v = pym.Signal("v", v.copy())

        s_out = pym.EinSum("ij,jk,k->ik")(s_A, s_B, s_v)
        s_out.tag = 'A.B.v'

        npt.assert_allclose(out_chk, s_out.state)
        pym.finite_difference(tosig=s_out, test_fn=self.fd_testfn, dx=1e-5)


    def test_scalar_constant(self):
        A, B = make_mat(100, 7), make_mat(100, 7)
        out_chk = np.einsum(',iB,iC->CB', 1e6, A, B)

        s_A = pym.Signal('A', A.copy())
        s_out = pym.EinSum(',iB,iC->CB')(1e6, s_A, B)
        s_out.tag = 'c.A.B'

        npt.assert_allclose(out_chk, s_out.state)
        pym.finite_difference(tosig=s_out, test_fn=self.fd_testfn, dx=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
