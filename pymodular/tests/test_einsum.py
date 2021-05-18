from unittest import TestCase

import pymodular as pym
import json
import numpy as np

def assert_fd(obj, x0, dx, dg_an, dg_fd):
    obj.assertAlmostEqual(dg_an, dg_fd)

# TODO:  dot product, outer product

class TestEinSum(TestCase):
    def test_einsum_vec_sum(self):
        n = 4

        a = np.random.rand(n)

        out_chk = np.sum(a)

        s_a = pym.Signal("vec", a)
        s_out = pym.Signal("sum(a)")

        blk = pym.EinSum(s_a, s_out, expression="i->")

        blk.response()
        self.assertTrue(np.allclose(out_chk, s_out.state))
        test_fn = lambda x0, dx, dg_an, dg_fd: assert_fd(self, x0, dx, dg_an, dg_fd)
        pym.finite_difference(blk, test_fn=test_fn)

    def test_einsum_mat_trace(self):
        n = 4

        a = np.random.rand(n, n)

        out_chk = np.trace(a)

        s_a = pym.Signal("mat", a)
        s_out = pym.Signal("trace(A)")

        blk = pym.EinSum(s_a, s_out, expression="ii->")

        blk.response()
        self.assertTrue(np.allclose(out_chk, s_out.state))
        test_fn = lambda x0, dx, dg_an, dg_fd: assert_fd(self, x0, dx, dg_an, dg_fd)
        pym.finite_difference(blk, test_fn=test_fn)

    def test_einsum_mat_sum(self):
        n = 3

        a = np.random.rand(n, n)

        out_chk = np.sum(a)

        s_a = pym.Signal("mat", a)
        s_out = pym.Signal("sum(A)")

        blk = pym.EinSum(s_a, s_out, expression="ij->")

        blk.response()
        self.assertTrue(np.allclose(out_chk, s_out.state))
        test_fn = lambda x0, dx, dg_an, dg_fd: assert_fd(self, x0, dx, dg_an, dg_fd)
        pym.finite_difference(blk, test_fn=test_fn)

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
    #     test_fn = lambda x0, dx, dg_an, dg_fd: assert_fd(self, x0, dx, dg_an, dg_fd)
    #     pym.finite_difference(blk, test_fn=test_fn)

    def test_einsum_matmat(self):
        n = 3

        a = np.random.rand(n, n)
        b = np.random.rand(n, n)

        out_chk = a.dot(b)

        s_a = pym.Signal("matA", a)
        s_b = pym.Signal("matB", b)
        s_out = pym.Signal("matC")

        blk = pym.EinSum([s_a, s_b], s_out, expression="ij,jk->ik")

        blk.response()
        self.assertTrue(np.allclose(out_chk, s_out.state))
        test_fn = lambda x0, dx, dg_an, dg_fd: assert_fd(self, x0, dx, dg_an, dg_fd)
        pym.finite_difference(blk, test_fn=test_fn)

    def test_einsum_matmatvec(self):
        N = 5
        p = 2

        A = np.random.rand(N, p)
        B = np.random.rand(p, p)
        v = np.random.rand(p)

        out_chk = (A.dot(B)).dot(np.diag(v))

        s_A = pym.Signal("A", A)
        s_B = pym.Signal("B", B)
        s_v = pym.Signal("v", v)
        s_out = pym.Signal("C")

        blk = pym.EinSum([s_A, s_B, s_v], s_out, expression="ij,jk,k->ik")

        blk.response()
        self.assertTrue(np.allclose(out_chk, s_out.state))
        test_fn = lambda x0, dx, dg_an, dg_fd: assert_fd(self, x0, dx, dg_an, dg_fd)
        pym.finite_difference(blk, test_fn=test_fn)

