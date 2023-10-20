import unittest

import pymoto as pym
import numpy as np


class TestEinSum(unittest.TestCase):
    def assert_fd(self, x0, dx, dg_an, dg_fd):
        self.assertAlmostEqual(dg_an, dg_fd, places=6)

    def test_vec_sum(self):
        n = 4

        a_list = [np.random.rand(n), np.random.rand(n)+1j*np.random.rand(n)]
        for a in a_list:
            out_chk = np.sum(a)

            s_a = pym.Signal("vec", a)
            s_out = pym.Signal("sum(a)")

            blk = pym.EinSum(s_a, s_out, expression="i->")

            blk.response()
            self.assertTrue(np.allclose(out_chk, s_out.state))
            pym.finite_difference(blk, test_fn=self.assert_fd)

    def test_vec_dot(self):
        n = 4

        a_list = [np.random.rand(n), np.random.rand(n)+1j*np.random.rand(n), np.random.rand(n)]
        b_list = [np.random.rand(n), np.random.rand(n)+1j*np.random.rand(n), np.random.rand(n)+1j*np.random.rand(n)]
        for a, b in zip(a_list, b_list):
            out_chk = a@b

            s_a = pym.Signal("veca", a)
            s_b = pym.Signal("vecb", b)
            s_out = pym.Signal("a^T.b")

            blk = pym.EinSum([s_a, s_b], s_out, expression="i,i->")

            blk.response()
            self.assertTrue(np.allclose(out_chk, s_out.state))
            pym.finite_difference(blk, test_fn=self.assert_fd)

    def test_vec_outer(self):
        n = 4

        a_list = [np.random.rand(n), np.random.rand(n)+1j*np.random.rand(n), np.random.rand(n)+1j*np.random.rand(n)]
        b_list = [np.random.rand(n), np.random.rand(n)+1j*np.random.rand(n), np.random.rand(n)]
        for a, b in zip(a_list, b_list):
            out_chk = np.outer(a, b)

            s_a = pym.Signal("veca", a)
            s_b = pym.Signal("vecb", b)
            s_out = pym.Signal("a.b^T")

            blk = pym.EinSum([s_a, s_b], s_out, expression="i,j->ij")

            blk.response()
            self.assertTrue(np.allclose(out_chk, s_out.state))
            pym.finite_difference(blk, test_fn=self.assert_fd)

    def test_mat_trace(self):
        n = 4

        a_list = [np.random.rand(n, n), np.random.rand(n, n)+1j*np.random.rand(n, n)]

        for a in a_list:
            out_chk = np.trace(a)

            s_a = pym.Signal("mat", a)
            s_out = pym.Signal("trace(A)")

            blk = pym.EinSum(s_a, s_out, expression="ii->")

            blk.response()
            self.assertTrue(np.allclose(out_chk, s_out.state))
            pym.finite_difference(blk, test_fn=self.assert_fd)

    def test_mat_sum(self):
        n = 3

        a_list = [np.random.rand(n, n), np.random.rand(n, n)+1j*np.random.rand(n, n)]
        for a in a_list:
            out_chk = np.sum(a)

            s_a = pym.Signal("mat", a)
            s_out = pym.Signal("sum(A)")

            blk = pym.EinSum(s_a, s_out, expression="ij->")

            blk.response()
            self.assertTrue(np.allclose(out_chk, s_out.state))
            pym.finite_difference(blk, test_fn=self.assert_fd)

    # TODO: Sensitivity does not work for this
    # def test_einsum_mat_diag(self):
    #     n = 3
    #
    #     a = np.random.rand(n, n)
    #
    #     out_chk = np.diag(a)
    #
    #     s_a = pym.Signal("mat", a)
    #     s_out = pym.Signal("diag(A)")
    #
    #     blk = pym.EinSum(s_a, s_out, expression="ii->i")
    #
    #     blk.response()
    #     self.assertTrue(np.allclose(out_chk, s_out.state))
    #     pym.finite_difference(blk, test_fn=self.assert_fd)

    def test_matmat(self):
        n = 3

        a_list = [np.random.rand(n, n), np.random.rand(n, n)+1j*np.random.rand(n, n), np.random.rand(n, n)]
        b_list = [np.random.rand(n, n), np.random.rand(n, n)+1j*np.random.rand(n, n), np.random.rand(n, n)+1j*np.random.rand(n, n)]
        for a, b in zip(a_list, b_list):
            out_chk = a.dot(b)

            s_a = pym.Signal("matA", a)
            s_b = pym.Signal("matB", b)
            s_out = pym.Signal("matC")

            blk = pym.EinSum([s_a, s_b], s_out, expression="ij,jk->ik")

            blk.response()
            self.assertTrue(np.allclose(out_chk, s_out.state))
            pym.finite_difference(blk, test_fn=self.assert_fd)

    def test_matmatvec(self):
        N = 5
        p = 2
        Areal = np.random.rand(N, p)
        Aimag = 1j*np.random.rand(N, p)
        Acplx = np.random.rand(N, p) + 1j*np.random.rand(N, p)

        Breal = np.random.rand(p, p)
        Bimag = 1j * np.random.rand(p, p)
        Bcplx = np.random.rand(p, p) + 1j * np.random.rand(p, p)

        vreal = np.random.rand(p)
        vimag = 1j * np.random.rand(p)
        vcplx = np.random.rand(p) + 1j * np.random.rand(p)
        ABv = [(Areal, Breal, vreal),
               (Aimag, Bimag, vimag),
               (Acplx, Bcplx, vcplx),
               (Areal, Bcplx, vimag),
               (Acplx, Breal, vreal)]

        for A, B, v in ABv:
            out_chk = (A.dot(B)).dot(np.diag(v))

            s_A = pym.Signal("A", A)
            s_B = pym.Signal("B", B)
            s_v = pym.Signal("v", v)
            s_out = pym.Signal("C")

            blk = pym.EinSum([s_A, s_B, s_v], s_out, expression="ij,jk,k->ik")

            blk.response()
            self.assertTrue(np.allclose(out_chk, s_out.state))
            pym.finite_difference(blk, test_fn=self.assert_fd)


if __name__ == '__main__':
    unittest.main()
