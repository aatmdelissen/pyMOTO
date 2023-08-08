import unittest
import pymoto as pym
import numpy as np
import scipy.sparse as spsp


def generate_random(*dn, lower=-1.0, upper=1.0):
    return np.random.rand(*dn) * (upper - lower) + lower


class TestDyadCarrier(unittest.TestCase):
    # flake8: noqa: C901
    @staticmethod
    def setup_dyads(n=10, complex=False, nonsquare=False, empty=True, rnd=generate_random):
        dyads = {}

        if empty:
            dyads['empty'] = pym.DyadCarrier()

        ndyads = [1, 2, 10]
        for s in ndyads:
            dyads[f'square_{s}_dyads'] = pym.DyadCarrier([rnd(n) for _ in range(s)], [rnd(n) for _ in range(s)])

        if complex:
            for s in ndyads:
                dyads[f'square_complex_{s}_dyads'] = \
                    pym.DyadCarrier([rnd(n)+1j*rnd(n) for _ in range(s)],
                                    [rnd(n)+1j*rnd(n) for _ in range(s)])

        if nonsquare:
            nonsquare_offsets_u = [1]
            nonsquare_offsets_v = [2]

            for off_u in nonsquare_offsets_u:
                for s in ndyads:
                    dyads[f'nonsquare_u_{s}_dyads'] = \
                        pym.DyadCarrier([rnd(n+off_u) for _ in range(s)],
                                        [rnd(n) for _ in range(s)])
            for off_v in nonsquare_offsets_v:
                for s in ndyads:
                    dyads[f'nonsquare_v_{s}_dyads'] = \
                        pym.DyadCarrier([rnd(n) for _ in range(s)],
                                        [rnd(n+off_v) for _ in range(s)])

            if complex:
                for off_u in nonsquare_offsets_u:
                    for s in ndyads:
                        dyads[f'nonsquare_u_complex_{s}_dyads'] = \
                            pym.DyadCarrier([rnd(n+off_u)+1j*rnd(n+off_u) for _ in range(s)],
                                            [rnd(n)+1j*rnd(n) for _ in range(s)])
                for off_v in nonsquare_offsets_v:
                    for s in ndyads:
                        dyads[f'nonsquare_v_complex_{s}_dyads'] = \
                            pym.DyadCarrier([rnd(n)+1j*rnd(n) for _ in range(s)],
                                            [rnd(n+off_v)+1j*rnd(n+off_v) for _ in range(s)])
        return dyads

    def test_initialize(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        dyad1 = pym.DyadCarrier(u1, v1)
        self.assertEqual(len(dyad1.u), 1)
        self.assertEqual(len(dyad1.v), 1)
        self.assertEqual(len(dyad1.u[0]), len(u1))
        self.assertEqual(len(dyad1.v[0]), len(v1))
        self.assertTrue(np.allclose(dyad1.u[0], u1))
        self.assertTrue(np.allclose(dyad1.v[0], v1))
        self.assertEqual(dyad1.shape, (10, 10))

        dyad2 = pym.DyadCarrier([u1, u2], [v1, v2])
        self.assertEqual(len(dyad2.u), 2)
        self.assertEqual(len(dyad2.v), 2)
        self.assertEqual(len(dyad2.u[0]), len(u1))
        self.assertEqual(len(dyad2.u[1]), len(u2))
        self.assertEqual(len(dyad2.v[0]), len(v1))
        self.assertEqual(len(dyad2.v[1]), len(v2))
        self.assertTrue(np.allclose(dyad2.u[0], u1))
        self.assertTrue(np.allclose(dyad2.u[1], u2))
        self.assertTrue(np.allclose(dyad2.v[0], v1))
        self.assertTrue(np.allclose(dyad2.v[1], v2))
        self.assertEqual(dyad2.shape, (10, 10))

        dyad3 = pym.DyadCarrier(np.random.rand(2, n), np.random.rand(2, n))
        self.assertEqual(len(dyad3.u), 1)
        self.assertEqual(len(dyad3.v), 1)
        self.assertEqual(dyad3.shape, (10, 10))

        dyad4 = pym.DyadCarrier(np.random.rand(2, n), np.random.rand(3, n))
        self.assertEqual(len(dyad4.u), 1)
        self.assertEqual(len(dyad4.v), 1)
        self.assertEqual(dyad4.shape, (10, 10))

        ue = np.random.rand(n)
        ve = np.random.rand(2, n)
        dyad5 = pym.DyadCarrier(ue, ve)
        self.assertEqual(len(dyad5.u), 1)
        self.assertEqual(len(dyad5.v), 1)
        self.assertTrue(np.allclose(dyad5.u[0], ue))
        self.assertTrue(np.allclose(dyad5.v[0], ve[0, :] + ve[1, :]))
        self.assertEqual(dyad5.shape, (10, 10))

        uf = np.random.rand(5)
        vf = np.random.rand(10)
        dyad6 = pym.DyadCarrier(uf, vf)
        self.assertEqual(dyad6.shape, (5, 10))

        dyad7 = pym.DyadCarrier()
        self.assertEqual(len(dyad7.u), 0)
        self.assertEqual(len(dyad7.v), 0)
        dyad7.add_dyad(u1)
        self.assertEqual(len(dyad7.u), 1)
        self.assertEqual(len(dyad7.v), 1)
        self.assertTrue(np.allclose(dyad7.u[0], u1))
        self.assertTrue(np.allclose(dyad7.v[0], u1))
        dyad7.add_dyad(u2, v2)
        self.assertEqual(len(dyad7.u), 2)
        self.assertEqual(len(dyad7.v), 2)
        self.assertTrue(np.allclose(dyad7.u[1], u2))
        self.assertTrue(np.allclose(dyad7.v[1], v2))

        dyad8 = pym.DyadCarrier(u1)
        self.assertEqual(len(dyad8.u), 1)
        self.assertEqual(len(dyad8.v), 1)
        self.assertTrue(np.allclose(dyad8.u[0], u1))
        self.assertTrue(np.allclose(dyad8.v[0], u1))

        u9_1 = np.random.rand(2, 3, n)
        dyad9 = pym.DyadCarrier(u9_1, v1)
        self.assertEqual(len(dyad9.u), 1)
        self.assertEqual(len(dyad9.v), 1)
        uchk = u9_1[0, 0, :] + u9_1[0, 1, :] + u9_1[0, 2, :] + u9_1[1, 0, :] + u9_1[1, 1, :] + u9_1[1, 2, :]
        self.assertTrue(np.allclose(dyad9.u[0], uchk))
        self.assertTrue(np.allclose(dyad9.v[0], v1))

        self.assertRaises(TypeError, pym.DyadCarrier, u1, [v1, v2])

        self.assertRaises(TypeError, pym.DyadCarrier, [u1, np.random.rand(n + 1)], [v1, v2])

        self.assertRaises(TypeError, pym.DyadCarrier, [u1, [1.0, 2.0, 3.0]], [v1, v2])

        self.assertRaises(TypeError, pym.DyadCarrier, [u1, u2], [v1, np.random.rand(n + 1)])

        self.assertRaises(TypeError, pym.DyadCarrier, [u1, u2], [v1, [1.0, 2.0, 3.0]])

    def test_math_operations(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadCarrier(u1, v1)
        b = pym.DyadCarrier([u1, u2], [v1, v2])

        ap = +a
        self.assertFalse(a.u[0] is ap.u[0])
        self.assertFalse(a.v[0] is ap.v[0])
        self.assertTrue(np.allclose(a.u[0], ap.u[0]))
        self.assertTrue(np.allclose(a.v[0], ap.v[0]))

        bm = -b
        self.assertFalse(b.u[0] is bm.u[0])
        self.assertFalse(b.u[1] is bm.u[1])
        self.assertFalse(b.v[0] is bm.v[0])
        self.assertFalse(b.v[1] is bm.v[1])
        self.assertTrue(np.allclose(b.u[0], -bm.u[0]))
        self.assertTrue(np.allclose(b.u[1], -bm.u[1]))
        self.assertTrue(np.allclose(b.v[0], bm.v[0]))
        self.assertTrue(np.allclose(b.v[1], bm.v[1]))

        ap += b
        self.assertEqual(len(ap.u), 3)
        self.assertEqual(len(ap.v), 3)
        self.assertTrue(np.allclose(ap.u[0], a.u[0]))
        self.assertTrue(np.allclose(ap.v[0], a.v[0]))
        self.assertTrue(np.allclose(ap.u[1], b.u[0]))
        self.assertTrue(np.allclose(ap.v[1], b.v[0]))
        self.assertTrue(np.allclose(ap.u[2], b.u[1]))
        self.assertTrue(np.allclose(ap.v[2], b.v[1]))

        a1 = +a
        a1 -= b
        self.assertEqual(len(a1.u), 3)
        self.assertEqual(len(a1.v), 3)
        self.assertTrue(np.allclose(a1.u[0], a.u[0]))
        self.assertTrue(np.allclose(a1.v[0], a.v[0]))
        self.assertTrue(np.allclose(a1.u[1], -b.u[0]))
        self.assertTrue(np.allclose(a1.v[1], b.v[0]))
        self.assertTrue(np.allclose(a1.u[2], -b.u[1]))
        self.assertTrue(np.allclose(a1.v[2], b.v[1]))

        a2 = a + b
        self.assertEqual(len(a2.u), 3)
        self.assertEqual(len(a2.v), 3)
        self.assertTrue(np.allclose(a2.u[0], a.u[0]))
        self.assertTrue(np.allclose(a2.v[0], a.v[0]))
        self.assertTrue(np.allclose(a2.u[1], b.u[0]))
        self.assertTrue(np.allclose(a2.v[1], b.v[0]))
        self.assertTrue(np.allclose(a2.u[2], b.u[1]))
        self.assertTrue(np.allclose(a2.v[2], b.v[1]))

        a3 = a - b
        self.assertEqual(len(a3.u), 3)
        self.assertEqual(len(a3.v), 3)
        self.assertTrue(np.allclose(a3.u[0], a.u[0]))
        self.assertTrue(np.allclose(a3.v[0], a.v[0]))
        self.assertTrue(np.allclose(a3.u[1], -b.u[0]))
        self.assertTrue(np.allclose(a3.v[1], b.v[0]))
        self.assertTrue(np.allclose(a3.u[2], -b.u[1]))
        self.assertTrue(np.allclose(a3.v[2], b.v[1]))

        a4 = a.copy()
        self.assertFalse(a.u[0] is a4.u[0])
        self.assertFalse(a.v[0] is a4.v[0])
        self.assertTrue(np.allclose(a4.u[0], a.u[0]))
        self.assertTrue(np.allclose(a4.v[0], a.v[0]))

    def test_complex(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)
        u1c = u1 + 1j * np.random.rand(n)
        u2c = u2 + 1j * np.random.rand(n)
        v1c = v1 + 1j * np.random.rand(n)
        v2c = v2 + 1j * np.random.rand(n)

        a = pym.DyadCarrier([u1c, u2], [v1, v2])
        b = pym.DyadCarrier([u1, u2c], [v1, v2c])
        c = pym.DyadCarrier([u1c, u2c], [v1c, v2c])
        d = pym.DyadCarrier([u1, u2], [v1, v2])

        self.assertTrue(a.iscomplex(), msg="DyadCarrier should be recognized as complex, even if 1 entry is complex")
        self.assertTrue(b.iscomplex(), msg="DyadCarrier should be recognized as complex, even if 1 entry is complex")
        self.assertTrue(c.iscomplex(), msg="DyadCarrier should be recognized as complex")
        self.assertFalse(d.iscomplex(), msg="DyadCarrier should be recognized as real")

        # Complex conjugations
        self.assertTrue(np.allclose(np.conj(a.todense()), a.conj().todense()))
        self.assertTrue(np.allclose(np.conj(b.todense()), b.conj().todense()))

        cconj = c.conj()
        self.assertTrue(np.allclose(np.conj(c.u[0]), cconj.u[0]))
        self.assertTrue(np.allclose(np.conj(c.u[1]), cconj.u[1]))
        self.assertTrue(np.allclose(np.conj(c.v[0]), cconj.v[0]))
        self.assertTrue(np.allclose(np.conj(c.v[1]), cconj.v[1]))
        self.assertTrue(np.allclose(np.conj(c.todense()), cconj.todense()))

        dconj = d.conj()
        self.assertTrue(np.allclose(np.conj(d.u[0]), dconj.u[0]))
        self.assertTrue(np.allclose(np.conj(d.u[1]), dconj.u[1]))
        self.assertTrue(np.allclose(np.conj(d.v[0]), dconj.v[0]))
        self.assertTrue(np.allclose(np.conj(d.v[1]), dconj.v[1]))
        self.assertTrue(np.allclose(np.conj(d.todense()), dconj.todense()))

        # Get real part
        self.assertTrue(np.allclose(np.real(a.todense()), a.real.todense()))
        self.assertTrue(np.allclose(np.real(b.todense()), b.real.todense()))
        self.assertTrue(np.allclose(np.real(c.todense()), c.real.todense()))
        self.assertTrue(np.allclose(np.real(d.todense()), d.real.todense()))

        # Get imaginary part
        self.assertTrue(np.allclose(np.imag(a.todense()), a.imag.todense()))
        self.assertTrue(np.allclose(np.imag(b.todense()), b.imag.todense()))
        self.assertTrue(np.allclose(np.imag(c.todense()), c.imag.todense()))
        self.assertTrue(np.allclose(np.imag(d.todense()), d.imag.todense()))

    def test_todense(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadCarrier([u1, u2], [v1, v2])

        achk = np.outer(u1, v1) + np.outer(u2, v2)
        self.assertTrue(np.allclose(achk, a.todense()))

        uu1 = np.random.rand(2, n)
        vv1 = np.random.rand(2, n)
        b = pym.DyadCarrier(uu1, vv1)
        bchk = np.outer(uu1[0, :], vv1[0, :]) + np.outer(uu1[1, :], vv1[0, :]) \
            + np.outer(uu1[0, :], vv1[1, :]) + np.outer(uu1[1, :], vv1[1, :])
        self.assertTrue(np.allclose(b.todense(), bchk))

        cu1 = np.random.rand(2, 2, n)
        cv1 = np.random.rand(2, 2, n)
        c = pym.DyadCarrier(cu1, cv1)
        cchk = np.outer(cu1[0, 0, :], cv1[0, 0, :]) + np.outer(cu1[0, 1, :], cv1[0, 0, :]) \
            + np.outer(cu1[0, 0, :], cv1[0, 1, :]) + np.outer(cu1[0, 1, :], cv1[0, 1, :]) \
            + np.outer(cu1[1, 0, :], cv1[0, 0, :]) + np.outer(cu1[1, 1, :], cv1[0, 0, :]) \
            + np.outer(cu1[1, 0, :], cv1[0, 1, :]) + np.outer(cu1[1, 1, :], cv1[0, 1, :]) \
            + np.outer(cu1[0, 0, :], cv1[1, 0, :]) + np.outer(cu1[0, 1, :], cv1[1, 0, :]) \
            + np.outer(cu1[0, 0, :], cv1[1, 1, :]) + np.outer(cu1[0, 1, :], cv1[1, 1, :]) \
            + np.outer(cu1[1, 0, :], cv1[1, 0, :]) + np.outer(cu1[1, 1, :], cv1[1, 0, :]) \
            + np.outer(cu1[1, 0, :], cv1[1, 1, :]) + np.outer(cu1[1, 1, :], cv1[1, 1, :])
        self.assertTrue(np.allclose(c.todense(), cchk))

        d = pym.DyadCarrier([u1, uu1], [vv1, v1])
        dchk = np.outer(u1, vv1[0, :]) + np.outer(u1, vv1[1, :]) + np.outer(uu1[0, :], v1) + np.outer(uu1[1, :], v1)
        self.assertTrue(np.allclose(d.todense(), dchk))

        large_dyad = pym.DyadCarrier(np.random.rand(4000))
        self.assertWarns(ResourceWarning, large_dyad.todense)

        empty_dyad = pym.DyadCarrier()
        self.assertEqual(empty_dyad.todense().shape, (0, 0))

        u3 = np.random.rand(15)
        u4 = np.random.rand(15)

        dyad5 = pym.DyadCarrier([u3, u4], [v1, v2])
        dyad5chk = np.outer(u3, v1) + np.outer(u4, v2)
        self.assertTrue(np.allclose(dyad5chk, dyad5.todense()))

    def test_todense_complex(self):
        n = 10
        u1 = np.random.rand(n) + 1j * np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n) + 1j * np.random.rand(n)
        v2 = np.random.rand(n) + 1j * np.random.rand(n)

        dyad1 = pym.DyadCarrier([u1, u2], [v1, v2])
        dyad1chk = np.outer(u1, v1) + np.outer(u2, v2)
        self.assertTrue(np.allclose(dyad1chk, dyad1.todense()))

        dyad2 = dyad1.conj()
        dyad2chk = np.conj(dyad1chk)
        self.assertTrue(np.allclose(dyad2chk, dyad2.todense()))

    def test_contract(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadCarrier([u1, u2], [v1, v2])

        tol = 1e-13
        self.assertAlmostEqual(a.contract(), np.dot(u1, v1) + np.dot(u2, v2), delta=tol)

        self.assertAlmostEqual(a.contract(np.eye(10)), np.dot(u1, v1) + np.dot(u2, v2), delta=tol)

        a_mat = np.random.rand(n, n)
        self.assertAlmostEqual(a.contract(a_mat), np.dot(u1, a_mat.dot(v1)) + np.dot(u2, a_mat.dot(v2)), delta=tol)

        a_submat = np.random.rand(3, 4)
        rows = np.array([3, 5, 5])
        cols = np.array([0, 1, 2, 8])

        self.assertAlmostEqual(a.contract(a_submat, rows, cols),
                               u1[rows].dot(a_submat.dot(v1[cols])) + u2[rows].dot(a_submat.dot(v2[cols])), delta=tol)
        self.assertAlmostEqual(a.contract(rows=cols, cols=cols),
                               np.dot(u1[cols], v1[cols]) + np.dot(u2[cols], v2[cols]), delta=tol)
        self.assertRaises(ValueError, a.contract, rows=rows, cols=cols)

        a_submat1 = np.random.rand(3, n)
        self.assertAlmostEqual(a.contract(a_submat1, rows),
                               u1[rows].dot(a_submat1.dot(v1)) + u2[rows].dot(a_submat1.dot(v2)), delta=tol)

        self.assertAlmostEqual(a.contract(a_submat1.T, cols=rows),
                               u1.dot(a_submat1.T.dot(v1[rows])) + u2.dot(a_submat1.T.dot(v2[rows])), delta=tol)

    def test_contract_batch(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadCarrier([u1, u2], [v1, v2])

        tol = 1e-13

        # Batch matrix
        a_mat = np.random.rand(3, n, n)
        self.assertEqual(a.contract(a_mat).shape, (3,))
        self.assertTrue(np.allclose(a.contract(a_mat), np.dot(u1, a_mat.dot(v1).T) + np.dot(u2, a_mat.dot(v2).T)))

        a_mat1 = np.random.rand(3, 4, n, n)
        self.assertEqual(a.contract(a_mat1).shape, (3, 4))
        self.assertTrue(np.allclose(a.contract(a_mat1), a_mat1.dot(v1).dot(u1) + a_mat1.dot(v2).dot(u2)))

        a_matfail = np.random.rand(n, n, 3, 4)
        self.assertRaises(ValueError, a.contract, a_matfail)

        # Batch rows
        a_submat = np.random.rand(3, 4)
        rows = np.array([[3, 5, 5], [5, 6, 7], [8, 9, 0]])
        cols = np.array([0, 1, 2, 8])
        self.assertEqual(a.contract(a_submat, rows, cols).shape, (3,))
        self.assertTrue(np.allclose(a.contract(a_submat, rows, cols),
                                    u1[rows].dot(a_submat.dot(v1[cols])) + u2[rows].dot(a_submat.dot(v2[cols]))))

        self.assertEqual(a.contract(rows=rows, cols=rows).shape, (3,))
        self.assertTrue(np.allclose(a.contract(rows=rows, cols=rows),
                                    np.sum(u1[rows] * v1[rows], axis=1) + np.sum(u2[rows] * v2[rows], axis=1)))

        cols_fail = np.array([[0, 1, 2, 8], [3, 4, 7, 8]])
        self.assertRaises(ValueError, a.contract, rows=rows, cols=cols_fail)

        # Batch cols
        a_submat1 = np.random.rand(3, 4)
        rows = np.array([3, 5, 5])
        cols = np.array([[0, 1, 2, 8], [3, 4, 7, 8], [5, 4, 1, 0]])
        self.assertEqual(a.contract(a_submat1, rows, cols).shape, (3,))
        self.assertTrue(np.allclose(a.contract(a_submat1, rows, cols),
                                    u1[rows].dot(a_submat1.dot(v1[cols].T)) + u2[rows].dot(a_submat1.dot(v2[cols].T))))

        # Batch all
        rows = np.array([[3, 5, 5], [5, 6, 7], [8, 9, 0]])
        cols = np.array([[0, 1, 2, 8], [3, 4, 7, 8], [5, 4, 1, 0]])
        a_submat2 = np.random.rand(3, 3, 4)
        self.assertAlmostEqual(a.contract(a_submat2, rows, cols)[0],
                               a.contract(a_submat2[0, :], rows[0, :], cols[0, :]), delta=tol)
        self.assertAlmostEqual(a.contract(a_submat2, rows, cols)[1],
                               a.contract(a_submat2[1, :], rows[1, :], cols[1, :]), delta=tol)
        self.assertAlmostEqual(a.contract(a_submat2, rows, cols)[2],
                               a.contract(a_submat2[2, :], rows[2, :], cols[2, :]), delta=tol)

        rows_fail = np.array([[3, 5, 5], [5, 6, 7]])
        self.assertRaises(ValueError, a.contract, a_submat2, rows_fail, cols)

    def test_contract_sparse(self):
        # Test contraction with a sparse matrix
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadCarrier([u1, u2], [v1, v2])
        diag = np.random.rand(1, n)
        S = spsp.spdiags(diag, 0, m=diag.size, n=diag.size)  # Size must be given for scipy==1.7

        self.assertAlmostEqual(a.contract(S), np.sum(u1 * diag * v1 + u2 * diag * v2), delta=1e-10)

    def test_contract_sparse_slice(self):
        # Test contraction with a sliced sparse matrix
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadCarrier([u1, u2], [v1, v2])
        diag = np.random.rand(1, n - 2)
        S = spsp.spdiags(diag, 0, m=diag.size, n=diag.size)

        self.assertAlmostEqual(a.contract(S, rows=np.arange(8), cols=np.arange(8)),
                               np.sum(u1[:8] * diag * v1[:8] + u2[:8] * diag * v2[:8]), delta=1e-10)

    def test_diagonal(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True)
        k_list = np.arange(-12, 12)
        for k in k_list:
            for key, d in dyads.items():
                self.assertTrue(np.allclose(d.diagonal(k), np.diagonal(d.todense(), offset=k)),
                                msg=f"Failed diagonal test with offset \"{k}\" and dyad \"{key}\"")

    def test_add_to_zeroarray(self):
        n = 10
        zer = np.array(0, dtype=object)
        zer += pym.DyadCarrier(np.random.rand(n), np.random.rand(n))
        self.assertTrue(isinstance(zer, pym.DyadCarrier))

    def test_dot(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            q = generate_random(d.shape[1])
            chk = d.todense().dot(q)
            self.assertTrue(np.allclose(d.dot(q), chk), msg=f"Failed dot test with dyad \"{key}\"")
            # np.dot(d, q) # This form cannot be overridden, so it will result in a list of dyadcarriers

    def test_transpose(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            chk = d.todense().T
            res1 = d.T.todense()
            self.assertTrue(np.allclose(chk, res1), msg=f"Failed transpose test with dyad \"{key}\"")

    def test_matmul_vec(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            q = generate_random(d.shape[1])
            chk = d.todense() @ q
            res = d @ q
            self.assertTrue(np.allclose(chk, res), msg=f"Failed matmul with dyad \"{key}\"")

    def test_rmatmul_vec(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            q = generate_random(d.shape[0])
            chk = q @ (d.todense())
            res = q @ d
            self.assertTrue(np.allclose(chk, res), msg=f"Failed matmul with dyad \"{key}\"")

    def test_matmul(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = generate_random(d.shape[1], d.shape[1])
            chk = d.todense() @ A
            res = (d @ A).todense()
            self.assertTrue(np.allclose(chk, res), msg=f"Failed matmul with dyad \"{key}\"")

    def test_rmatmul(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = generate_random(d.shape[0], d.shape[0])
            chk = A @ (d.todense())
            res = (A @ d).todense()
            self.assertTrue(np.allclose(chk, res), msg=f"Failed matmul with dyad \"{key}\"")

    def test_matmul_dyad(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = d.todense()
            chk = A.T @ A
            res = (d.T @ d).todense()
            self.assertTrue(np.allclose(chk, res), msg=f"Failed matmul with 2 dyads \"{key}\"")

    def test_slice(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = d.todense()

            self.assertTrue(np.allclose(A[0:3, :], d[0:3, :].todense()))
            self.assertTrue(np.allclose(A[0, :], d[0, :]))
            self.assertTrue(np.allclose(A[:, 1], d[:, 1]))
            self.assertTrue(np.allclose(A[0:5, 1:3], d[0:5, 1:3].todense()))
            self.assertTrue(np.allclose(A[0, 1], d[0, 1]))
            self.assertTrue(np.allclose(A[3, 1], d[3, 1]))

            indi = np.arange(0, 4)

            self.assertTrue(np.allclose(A[indi, :], d[indi, :].todense()))
            self.assertTrue(np.allclose(A[:, indi], d[:, indi].todense()))
            self.assertTrue(np.allclose(A[indi, indi], d[indi, indi]))

            indi = np.array([1, 1, 2, 3, 3])
            self.assertTrue(np.allclose(A[indi, :], d[indi, :].todense()))
            self.assertTrue(np.allclose(A[:, indi], d[:, indi].todense()))
            self.assertTrue(np.allclose(A[indi, indi], d[indi, indi]))

            self.assertRaises(IndexError, lambda: d[np.array([1, 2]), np.array([1, 2, 3])])

            indi = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
            indj = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
            self.assertTrue(np.allclose(A[indi, indj], d[indi, indj]))

    def test_slice_asignment(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            d0 = d.copy()
            A = d0.todense()
            A[0:3, :] = 0.0
            d[0:3, :] = 0.0
            self.assertTrue(np.allclose(A, d.todense()))

            A, d = d0.todense(), d0.copy()
            A[0, :] = 0.0
            d[0, :] = 0.0
            self.assertTrue(np.allclose(A, d.todense()))

            A, d = d0.todense(), d0.copy()
            A[:, 1] = 0.0
            d[:, 1] = 0.0
            self.assertTrue(np.allclose(A, d.todense()))

            # "Impossible" options for a low-rank matrix
            # A, d = d0.todense(), d0.copy()
            # A[0:5, 1:3] = 0.0
            # d[0:5, 1:3] = 0.0
            # self.assertTrue(np.allclose(A, d.todense()))

            # A, d = d0.todense(), d0.copy()
            # A[0, 1] = 0.0
            # d[0, 1] = 0.0
            # self.assertTrue(np.allclose(A, d.todense()))

            # A, d = d0.todense(), d0.copy()
            # A[3, 1] = 0.0
            # d[3, 1] = 0.0
            # self.assertTrue(np.allclose(A, d.todense()))

            indi = np.arange(0, 4)

            A, d = d0.todense(), d0.copy()
            A[indi, :] = 0.0
            d[indi, :] = 0.0
            self.assertTrue(np.allclose(A, d.todense()))

            A, d = d0.todense(), d0.copy()
            A[:, indi] = 0.0
            d[:, indi] = 0.0
            self.assertTrue(np.allclose(A, d.todense()))

            # self.assertTrue(np.allclose(A[indi, indi], d[indi, indi]))

            indi = np.array([1, 1, 2, 3, 3])
            A, d = d0.todense(), d0.copy()
            A[indi, :] = 0.0
            d[indi, :] = 0.0
            self.assertTrue(np.allclose(A, d.todense()))

            A, d = d0.todense(), d0.copy()
            A[:, indi] = 0.0
            d[:, indi] = 0.0
            self.assertTrue(np.allclose(A, d.todense()))
            # self.assertTrue(np.allclose(A[indi, indi], d[indi, indi]))

    def test_mul_with_scalar(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = d.todense()

            if np.iscomplexobj(A):
                c = generate_random(1) + 1j * generate_random(1)
            else:
                c = generate_random(1)

            chk = A * c
            res = (d * c).todense()
            self.assertTrue(np.allclose(chk, res))

            chk = A * c[0]
            res = (d * c[0]).todense()
            self.assertTrue(np.allclose(chk, res))

            chk = c * A
            res = (c * d).todense()
            self.assertTrue(np.allclose(chk, res))

            chk = c[0] * A
            res = (c[0] * d).todense()
            self.assertTrue(np.allclose(chk, res))


if __name__ == '__main__':
    unittest.main()
