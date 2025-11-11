import pytest
import pymoto as pym
import numpy as np
import numpy.testing as npt
import scipy.sparse as spsp


def generate_random(*dn, lower=-1.0, upper=1.0):
    return np.random.rand(*dn) * (upper - lower) + lower


class TestDyadicMatrix:
    # flake8: noqa: C901
    @staticmethod
    def setup_dyads(n=10, complex=False, nonsquare=False, empty=True, rnd=generate_random):
        dyads = {}

        if empty:
            dyads['empty'] = pym.DyadicMatrix()

        ndyads = [1, 2, 10]
        for s in ndyads:
            dyads[f'square_{s}_dyads'] = pym.DyadicMatrix([rnd(n) for _ in range(s)], [rnd(n) for _ in range(s)])

        if complex:
            for s in ndyads:
                dyads[f'square_complex_{s}_dyads'] = \
                    pym.DyadicMatrix([rnd(n)+1j*rnd(n) for _ in range(s)],
                                    [rnd(n)+1j*rnd(n) for _ in range(s)])

        if nonsquare:
            nonsquare_offsets_u = [1]
            nonsquare_offsets_v = [2]

            for off_u in nonsquare_offsets_u:
                for s in ndyads:
                    dyads[f'nonsquare_u_{s}_dyads'] = \
                        pym.DyadicMatrix([rnd(n+off_u) for _ in range(s)],
                                        [rnd(n) for _ in range(s)])
            for off_v in nonsquare_offsets_v:
                for s in ndyads:
                    dyads[f'nonsquare_v_{s}_dyads'] = \
                        pym.DyadicMatrix([rnd(n) for _ in range(s)],
                                        [rnd(n+off_v) for _ in range(s)])

            if complex:
                for off_u in nonsquare_offsets_u:
                    for s in ndyads:
                        dyads[f'nonsquare_u_complex_{s}_dyads'] = \
                            pym.DyadicMatrix([rnd(n+off_u)+1j*rnd(n+off_u) for _ in range(s)],
                                            [rnd(n)+1j*rnd(n) for _ in range(s)])
                for off_v in nonsquare_offsets_v:
                    for s in ndyads:
                        dyads[f'nonsquare_v_complex_{s}_dyads'] = \
                            pym.DyadicMatrix([rnd(n)+1j*rnd(n) for _ in range(s)],
                                            [rnd(n+off_v)+1j*rnd(n+off_v) for _ in range(s)])
        return dyads

    def test_initialize(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        dyad1 = pym.DyadicMatrix(u1, v1)
        assert len(dyad1.u) == 1
        assert len(dyad1.v) == 1
        assert len(dyad1.u[0]) == len(u1)
        assert len(dyad1.v[0]) == len(v1)
        assert np.allclose(dyad1.u[0], u1)
        assert np.allclose(dyad1.v[0], v1)
        assert dyad1.shape == (10, 10)

        dyad2 = pym.DyadicMatrix([u1, u2], [v1, v2])
        assert len(dyad2.u) == 2
        assert len(dyad2.v) == 2
        assert len(dyad2.u[0]) == len(u1)
        assert len(dyad2.u[1]) == len(u2)
        assert len(dyad2.v[0]) == len(v1)
        assert len(dyad2.v[1]) == len(v2)
        npt.assert_allclose(dyad2.u[0], u1)
        npt.assert_allclose(dyad2.u[1], u2)
        npt.assert_allclose(dyad2.v[0], v1)
        npt.assert_allclose(dyad2.v[1], v2)
        assert dyad2.shape == (10, 10)

        dyad3 = pym.DyadicMatrix(np.random.rand(2, n), np.random.rand(2, n))
        assert len(dyad3.u) == 1
        assert len(dyad3.v) == 1
        assert dyad3.shape == (10, 10)

        dyad4 = pym.DyadicMatrix(np.random.rand(2, n), np.random.rand(3, n))
        assert len(dyad4.u) == 1
        assert len(dyad4.v) == 1
        assert dyad4.shape == (10, 10)

        ue = np.random.rand(n)
        ve = np.random.rand(2, n)
        dyad5 = pym.DyadicMatrix(ue, ve)
        assert len(dyad5.u) == 1
        assert len(dyad5.v) == 1
        npt.assert_allclose(dyad5.u[0], ue)
        npt.assert_allclose(dyad5.v[0], ve[0, :] + ve[1, :])
        assert dyad5.shape == (10, 10)

        uf = np.random.rand(5)
        vf = np.random.rand(10)
        dyad6 = pym.DyadicMatrix(uf, vf)
        assert dyad6.shape == (5, 10)

        dyad7 = pym.DyadicMatrix()
        assert len(dyad7.u) == 0
        assert len(dyad7.v) == 0
        dyad7.add_dyad(u1)
        assert len(dyad7.u) == 1
        assert len(dyad7.v) == 1
        npt.assert_allclose(dyad7.u[0], u1)
        npt.assert_allclose(dyad7.v[0], u1)
        dyad7.add_dyad(u2, v2)
        assert len(dyad7.u) == 2
        assert len(dyad7.v) == 2
        npt.assert_allclose(dyad7.u[1], u2)
        npt.assert_allclose(dyad7.v[1], v2)

        # with shape
        dyad7a = pym.DyadicMatrix(shape=(n, n))
        assert dyad7a.shape == (n, n)
        dyad7.add_dyad(u1)
        assert dyad7a.shape == (n, n)
        pytest.raises(TypeError, dyad7.add_dyad, uf, vf)

        dyad8 = pym.DyadicMatrix(u1)
        assert len(dyad8.u) == 1
        assert len(dyad8.v) == 1
        npt.assert_allclose(dyad8.u[0], u1)
        npt.assert_allclose(dyad8.v[0], u1)

        u9_1 = np.random.rand(2, 3, n)
        dyad9 = pym.DyadicMatrix(u9_1, v1)
        assert len(dyad9.u) == 1
        assert len(dyad9.v) == 1
        uchk = u9_1[0, 0, :] + u9_1[0, 1, :] + u9_1[0, 2, :] + u9_1[1, 0, :] + u9_1[1, 1, :] + u9_1[1, 2, :]
        npt.assert_allclose(dyad9.u[0], uchk)
        npt.assert_allclose(dyad9.v[0], v1)

        pytest.raises(TypeError, pym.DyadicMatrix, u1, [v1, v2])

        pytest.raises(TypeError, pym.DyadicMatrix, [u1, np.random.rand(n + 1)], [v1, v2])

        pytest.raises(TypeError, pym.DyadicMatrix, [u1, [1.0, 2.0, 3.0]], [v1, v2])

        pytest.raises(TypeError, pym.DyadicMatrix, [u1, u2], [v1, np.random.rand(n + 1)])

        pytest.raises(TypeError, pym.DyadicMatrix, [u1, u2], [v1, [1.0, 2.0, 3.0]])

    def test_math_operations(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadicMatrix(u1, v1)
        b = pym.DyadicMatrix([u1, u2], [v1, v2])

        ap = +a
        assert a.u[0] is not ap.u[0]
        assert a.v[0] is not ap.v[0]
        npt.assert_allclose(a.u[0], ap.u[0])
        npt.assert_allclose(a.v[0], ap.v[0])

        bm = -b
        assert b.u[0] is not bm.u[0]
        assert b.u[1] is not bm.u[1]
        assert b.v[0] is not bm.v[0]
        assert b.v[1] is not bm.v[1]
        npt.assert_allclose(b.u[0], -bm.u[0])
        npt.assert_allclose(b.u[1], -bm.u[1])
        npt.assert_allclose(b.v[0], bm.v[0])
        npt.assert_allclose(b.v[1], bm.v[1])

        ap += b
        assert len(ap.u) == 3
        assert len(ap.v) == 3
        npt.assert_allclose(ap.u[0], a.u[0])
        npt.assert_allclose(ap.v[0], a.v[0])
        npt.assert_allclose(ap.u[1], b.u[0])
        npt.assert_allclose(ap.v[1], b.v[0])
        npt.assert_allclose(ap.u[2], b.u[1])
        npt.assert_allclose(ap.v[2], b.v[1])

        a1 = +a
        a1 -= b
        assert len(a1.u) == 3
        assert len(a1.v) == 3
        npt.assert_allclose(a1.u[0], a.u[0])
        npt.assert_allclose(a1.v[0], a.v[0])
        npt.assert_allclose(a1.u[1], -b.u[0])
        npt.assert_allclose(a1.v[1], b.v[0])
        npt.assert_allclose(a1.u[2], -b.u[1])
        npt.assert_allclose(a1.v[2], b.v[1])

        a2 = a + b
        assert len(a2.u) == 3
        assert len(a2.v) == 3
        npt.assert_allclose(a2.u[0], a.u[0])
        npt.assert_allclose(a2.v[0], a.v[0])
        npt.assert_allclose(a2.u[1], b.u[0])
        npt.assert_allclose(a2.v[1], b.v[0])
        npt.assert_allclose(a2.u[2], b.u[1])
        npt.assert_allclose(a2.v[2], b.v[1])

        a3 = a - b
        assert len(a3.u) == 3
        assert len(a3.v) == 3
        npt.assert_allclose(a3.u[0], a.u[0])
        npt.assert_allclose(a3.v[0], a.v[0])
        npt.assert_allclose(a3.u[1], -b.u[0])
        npt.assert_allclose(a3.v[1], b.v[0])
        npt.assert_allclose(a3.u[2], -b.u[1])
        npt.assert_allclose(a3.v[2], b.v[1])

        a4 = a.copy()
        assert a.u[0] is not a4.u[0]
        assert a.v[0] is not a4.v[0]
        npt.assert_allclose(a4.u[0], a.u[0])
        npt.assert_allclose(a4.v[0], a.v[0])

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

        a = pym.DyadicMatrix([u1c, u2], [v1, v2])
        b = pym.DyadicMatrix([u1, u2c], [v1, v2c])
        c = pym.DyadicMatrix([u1c, u2c], [v1c, v2c])
        d = pym.DyadicMatrix([u1, u2], [v1, v2])

        assert a.iscomplex(), "DyadicMatrix should be recognized as complex, even if 1 entry is complex"
        assert b.iscomplex(), "DyadicMatrix should be recognized as complex, even if 1 entry is complex"
        assert c.iscomplex(), "DyadicMatrix should be recognized as complex"
        assert not d.iscomplex(), "DyadicMatrix should be recognized as real"

        # Complex conjugations
        npt.assert_allclose(np.conj(a.todense()), a.conj().todense())
        npt.assert_allclose(np.conj(b.todense()), b.conj().todense())

        cconj = c.conj()
        npt.assert_allclose(np.conj(c.u[0]), cconj.u[0])
        npt.assert_allclose(np.conj(c.u[1]), cconj.u[1])
        npt.assert_allclose(np.conj(c.v[0]), cconj.v[0])
        npt.assert_allclose(np.conj(c.v[1]), cconj.v[1])
        npt.assert_allclose(np.conj(c.todense()), cconj.todense())

        dconj = d.conj()
        npt.assert_allclose(np.conj(d.u[0]), dconj.u[0])
        npt.assert_allclose(np.conj(d.u[1]), dconj.u[1])
        npt.assert_allclose(np.conj(d.v[0]), dconj.v[0])
        npt.assert_allclose(np.conj(d.v[1]), dconj.v[1])
        npt.assert_allclose(np.conj(d.todense()), dconj.todense())

        # Get real part
        npt.assert_allclose(np.real(a.todense()), a.real.todense())
        npt.assert_allclose(np.real(b.todense()), b.real.todense())
        npt.assert_allclose(np.real(c.todense()), c.real.todense())
        npt.assert_allclose(np.real(d.todense()), d.real.todense())

        # Get imaginary part
        npt.assert_allclose(np.imag(a.todense()), a.imag.todense())
        npt.assert_allclose(np.imag(b.todense()), b.imag.todense())
        npt.assert_allclose(np.imag(c.todense()), c.imag.todense())
        npt.assert_allclose(np.imag(d.todense()), d.imag.todense())

    def test_todense(self):
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadicMatrix([u1, u2], [v1, v2])

        achk = np.outer(u1, v1) + np.outer(u2, v2)
        npt.assert_allclose(achk, a.todense())

        uu1 = np.random.rand(2, n)
        vv1 = np.random.rand(2, n)
        b = pym.DyadicMatrix(uu1, vv1)
        bchk = np.outer(uu1[0, :], vv1[0, :]) + np.outer(uu1[1, :], vv1[0, :]) \
            + np.outer(uu1[0, :], vv1[1, :]) + np.outer(uu1[1, :], vv1[1, :])
        npt.assert_allclose(b.todense(), bchk)

        cu1 = np.random.rand(2, 2, n)
        cv1 = np.random.rand(2, 2, n)
        c = pym.DyadicMatrix(cu1, cv1)
        cchk = np.outer(cu1[0, 0, :], cv1[0, 0, :]) + np.outer(cu1[0, 1, :], cv1[0, 0, :]) \
            + np.outer(cu1[0, 0, :], cv1[0, 1, :]) + np.outer(cu1[0, 1, :], cv1[0, 1, :]) \
            + np.outer(cu1[1, 0, :], cv1[0, 0, :]) + np.outer(cu1[1, 1, :], cv1[0, 0, :]) \
            + np.outer(cu1[1, 0, :], cv1[0, 1, :]) + np.outer(cu1[1, 1, :], cv1[0, 1, :]) \
            + np.outer(cu1[0, 0, :], cv1[1, 0, :]) + np.outer(cu1[0, 1, :], cv1[1, 0, :]) \
            + np.outer(cu1[0, 0, :], cv1[1, 1, :]) + np.outer(cu1[0, 1, :], cv1[1, 1, :]) \
            + np.outer(cu1[1, 0, :], cv1[1, 0, :]) + np.outer(cu1[1, 1, :], cv1[1, 0, :]) \
            + np.outer(cu1[1, 0, :], cv1[1, 1, :]) + np.outer(cu1[1, 1, :], cv1[1, 1, :])
        npt.assert_allclose(c.todense(), cchk)

        d = pym.DyadicMatrix([u1, uu1], [vv1, v1])
        dchk = np.outer(u1, vv1[0, :]) + np.outer(u1, vv1[1, :]) + np.outer(uu1[0, :], v1) + np.outer(uu1[1, :], v1)
        npt.assert_allclose(d.todense(), dchk)

        large_dyad = pym.DyadicMatrix(np.random.rand(4000))
        pytest.warns(ResourceWarning, large_dyad.todense)

        empty_dyad = pym.DyadicMatrix()
        assert empty_dyad.todense().shape == (0, 0)

        u3 = np.random.rand(15)
        u4 = np.random.rand(15)

        dyad5 = pym.DyadicMatrix([u3, u4], [v1, v2])
        dyad5chk = np.outer(u3, v1) + np.outer(u4, v2)
        npt.assert_allclose(dyad5chk, dyad5.todense())

    def test_todense_complex(self):
        n = 10
        u1 = np.random.rand(n) + 1j * np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n) + 1j * np.random.rand(n)
        v2 = np.random.rand(n) + 1j * np.random.rand(n)

        dyad1 = pym.DyadicMatrix([u1, u2], [v1, v2])
        dyad1chk = np.outer(u1, v1) + np.outer(u2, v2)
        npt.assert_allclose(dyad1chk, dyad1.todense())

        dyad2 = dyad1.conj()
        dyad2chk = np.conj(dyad1chk)
        npt.assert_allclose(dyad2chk, dyad2.todense())


    def test_contract_sparse(self):
        # Test contraction with a sparse matrix
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadicMatrix([u1, u2], [v1, v2])
        diag = np.random.rand(1, n)
        S = spsp.spdiags(diag, 0, m=diag.size, n=diag.size)  # Size must be given for scipy==1.7

        npt.assert_allclose(a.contract(S), np.sum(u1 * diag * v1 + u2 * diag * v2))

    def test_contract_sparse_performance(self):
        # Test contraction with a sparse matrix
        np.random.seed(0)
        n = 10000

        a = pym.DyadicMatrix([np.random.rand(n) for _ in range(30)], [np.random.rand(n) for _ in range(30)])
        # diag = np.random.rand(1, n)
        # S = spsp.random(n, n, density=200/(n*n)).tocsc()  # Size must be given for scipy==1.7
        # for i in range(100):
        #     a.contract(S)

        # Generate random matrices
        mats = []
        for _ in range(500):
            N = np.random.randint(10, 200)
            rows = np.random.randint(0, n, N)
            cols = np.random.randint(0, n, N)
            data = np.random.rand(N)
            mats.append(spsp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocoo())

        for _ in range(10):
            a.contract_multi(mats)

        # npt.assert_allclose(a.contract(S), np.sum(u1 * diag * v1 + u2 * diag * v2))

    def test_contract_sparse_slice(self):
        # Test contraction with a sliced sparse matrix
        n = 10
        u1 = np.random.rand(n)
        u2 = np.random.rand(n)
        v1 = np.random.rand(n)
        v2 = np.random.rand(n)

        a = pym.DyadicMatrix([u1, u2], [v1, v2])
        diag = np.random.rand(1, n - 2)
        S = spsp.spdiags(diag, 0, m=diag.size, n=diag.size)

        npt.assert_allclose(a.contract(S, rows=np.arange(8), cols=np.arange(8)),
                            np.sum(u1[:8] * diag * v1[:8] + u2[:8] * diag * v2[:8]))

    def test_diagonal(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True)
        k_list = np.arange(-12, 12)
        for k in k_list:
            for key, d in dyads.items():
                msg=f"Failed diagonal test with offset \"{k}\" and dyad \"{key}\""
                npt.assert_allclose(d.diagonal(k), np.diagonal(d.todense(), offset=k), err_msg=msg)

    def test_add_to_zeroarray(self):
        n = 10
        zer = np.array(0, dtype=object)
        zer += pym.DyadicMatrix(np.random.rand(n), np.random.rand(n))
        assert isinstance(zer, pym.DyadicMatrix)

    def test_dot(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            q = generate_random(d.shape[1])
            chk = d.todense().dot(q)
            npt.assert_allclose(d.dot(q), chk, err_msg=f"Failed dot test with dyad \"{key}\"")
            # np.dot(d, q) # This form cannot be overridden, so it will result in a list of dyadcarriers

    def test_transpose(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            chk = d.todense().T
            res1 = d.T.todense()
            npt.assert_allclose(chk, res1, err_msg=f"Failed transpose test with dyad \"{key}\"")

    def test_matmul_vec(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            q = generate_random(d.shape[1])
            chk = d.todense() @ q
            res = d @ q
            npt.assert_allclose(chk, res, err_msg=f"Failed matmul with dyad \"{key}\"")

    def test_rmatmul_vec(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            q = generate_random(d.shape[0])
            chk = q @ (d.todense())
            res = q @ d
            npt.assert_allclose(chk, res, err_msg=f"Failed matmul with dyad \"{key}\"")

    def test_matmul(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = generate_random(d.shape[1], d.shape[1])
            chk = d.todense() @ A
            res = (d @ A).todense()
            npt.assert_allclose(chk, res, err_msg=f"Failed matmul with dyad \"{key}\"")

    def test_rmatmul(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = generate_random(d.shape[0], d.shape[0])
            chk = A @ (d.todense())
            res = (A @ d).todense()
            npt.assert_allclose(chk, res, err_msg=f"Failed matmul with dyad \"{key}\"")

    def test_matmul_dyad(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = d.todense()
            chk = A.T @ A
            res = (d.T @ d).todense()
            npt.assert_allclose(chk, res, err_msg=f"Failed matmul with 2 dyads \"{key}\"")

    def test_slice(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            A = d.todense()

            npt.assert_allclose(A[0:3, :], d[0:3, :].todense())
            npt.assert_allclose(A[0, :], d[0, :])
            npt.assert_allclose(A[:, 1], d[:, 1])
            npt.assert_allclose(A[0:5, 1:3], d[0:5, 1:3].todense())
            npt.assert_allclose(A[0, 1], d[0, 1])
            npt.assert_allclose(A[3, 1], d[3, 1])

            indi = np.arange(0, 4)

            npt.assert_allclose(A[indi, :], d[indi, :].todense())
            npt.assert_allclose(A[:, indi], d[:, indi].todense())
            npt.assert_allclose(A[indi, indi], d[indi, indi])

            indi = np.array([1, 1, 2, 3, 3])
            npt.assert_allclose(A[indi, :], d[indi, :].todense())
            npt.assert_allclose(A[:, indi], d[:, indi].todense())
            npt.assert_allclose(A[indi, indi], d[indi, indi])

            pytest.raises(IndexError, lambda: d[np.array([1, 2]), np.array([1, 2, 3])])

            indi = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
            indj = np.array([[1, 1, 1, 1], [2, 2, 2, 2]])
            npt.assert_allclose(A[indi, indj], d[indi, indj])

    def test_slice_asignment(self):
        n = 10
        dyads = self.setup_dyads(n, complex=True, nonsquare=True, empty=False)
        for key, d in dyads.items():
            d0 = d.copy()
            A = d0.todense()
            A[0:3, :] = 0.0
            d[0:3, :] = 0.0
            npt.assert_allclose(A, d.todense())

            A, d = d0.todense(), d0.copy()
            A[0, :] = 0.0
            d[0, :] = 0.0
            npt.assert_allclose(A, d.todense())

            A, d = d0.todense(), d0.copy()
            A[:, 1] = 0.0
            d[:, 1] = 0.0
            npt.assert_allclose(A, d.todense())

            # "Impossible" options for a low-rank matrix
            # A, d = d0.todense(), d0.copy()
            # A[0:5, 1:3] = 0.0
            # d[0:5, 1:3] = 0.0
            # npt.assert_allclose(A, d.todense())

            # A, d = d0.todense(), d0.copy()
            # A[0, 1] = 0.0
            # d[0, 1] = 0.0
            # npt.assert_allclose(A, d.todense())

            # A, d = d0.todense(), d0.copy()
            # A[3, 1] = 0.0
            # d[3, 1] = 0.0
            # npt.assert_allclose(A, d.todense())

            indi = np.arange(0, 4)

            A, d = d0.todense(), d0.copy()
            A[indi, :] = 0.0
            d[indi, :] = 0.0
            npt.assert_allclose(A, d.todense())

            A, d = d0.todense(), d0.copy()
            A[:, indi] = 0.0
            d[:, indi] = 0.0
            npt.assert_allclose(A, d.todense())

            # npt.assert_allclose(A[indi, indi], d[indi, indi]))

            indi = np.array([1, 1, 2, 3, 3])
            A, d = d0.todense(), d0.copy()
            A[indi, :] = 0.0
            d[indi, :] = 0.0
            npt.assert_allclose(A, d.todense())

            A, d = d0.todense(), d0.copy()
            A[:, indi] = 0.0
            d[:, indi] = 0.0
            npt.assert_allclose(A, d.todense())
            # npt.assert_allclose(A[indi, indi], d[indi, indi]))

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
            npt.assert_allclose(chk, res)

            chk = A * c[0]
            res = (d * c[0]).todense()
            npt.assert_allclose(chk, res)

            chk = c * A
            res = (c * d).todense()
            npt.assert_allclose(chk, res)

            chk = c[0] * A
            res = (c[0] * d).todense()
            npt.assert_allclose(chk, res)


@pytest.mark.parametrize("vec_complex", [True, False])
@pytest.mark.parametrize("mat_complex", [True, False])
def test_contract(vec_complex, mat_complex):
    np.random.seed(0)
    n = 10

    vc = 1j if vec_complex else 0
    mc = 1j if mat_complex else 0

    u1 = np.random.rand(n) + vc * np.random.rand(n)
    u2 = np.random.rand(n) + vc * np.random.rand(n)
    v1 = np.random.rand(n) + vc * np.random.rand(n)
    v2 = np.random.rand(n) + vc * np.random.rand(n)

    a = pym.DyadicMatrix([u1, u2], [v1, v2])

    npt.assert_allclose(a.contract(), np.dot(u1, v1) + np.dot(u2, v2))

    npt.assert_allclose(a.contract(np.eye(10)), np.dot(u1, v1) + np.dot(u2, v2))

    a_mat = np.random.rand(n, n) + mc * np.random.rand(n, n)
    npt.assert_allclose(a.contract(a_mat), np.dot(u1, a_mat.dot(v1)) + np.dot(u2, a_mat.dot(v2)))

    a_submat = np.random.rand(3, 4) + mc * np.random.rand(3, 4)
    rows = np.array([3, 5, 5])
    cols = np.array([0, 1, 2, 8])

    npt.assert_allclose(a.contract(a_submat, rows, cols),
                        u1[rows].dot(a_submat.dot(v1[cols])) + u2[rows].dot(a_submat.dot(v2[cols])))
    npt.assert_allclose(a.contract(rows=cols, cols=cols),
                        np.dot(u1[cols], v1[cols]) + np.dot(u2[cols], v2[cols]))
    pytest.raises(ValueError, a.contract, rows=rows, cols=cols)

    a_submat1 = np.random.rand(3, n) + mc * np.random.rand(3, n)
    npt.assert_allclose(a.contract(a_submat1, rows),
                        u1[rows].dot(a_submat1.dot(v1)) + u2[rows].dot(a_submat1.dot(v2)))

    npt.assert_allclose(a.contract(a_submat1.T, cols=rows),
                        u1.dot(a_submat1.T.dot(v1[rows])) + u2.dot(a_submat1.T.dot(v2[rows])))


@pytest.mark.parametrize("vec_complex", [True, False])
@pytest.mark.parametrize("mat_complex", [True, False])
def test_contract_batch(vec_complex, mat_complex):
    np.random.seed(0)
    n = 10
    vc = 1j if vec_complex else 0
    mc = 1j if mat_complex else 0

    u1 = np.random.rand(n) + vc * np.random.rand(n)
    u2 = np.random.rand(n) + vc * np.random.rand(n)
    v1 = np.random.rand(n) + vc * np.random.rand(n)
    v2 = np.random.rand(n) + vc * np.random.rand(n)

    a = pym.DyadicMatrix([u1, u2], [v1, v2])

    # Batch matrix
    a_mat = np.random.rand(3, n, n) + mc * np.random.rand(3, n, n)
    assert a.contract(a_mat).shape == (3,)
    npt.assert_allclose(a.contract(a_mat), np.dot(u1, a_mat.dot(v1).T) + np.dot(u2, a_mat.dot(v2).T))

    a_mat1 = np.random.rand(3, 4, n, n) + mc * np.random.rand(3, 4, n, n)
    assert a.contract(a_mat1).shape == (3, 4)
    npt.assert_allclose(a.contract(a_mat1), a_mat1.dot(v1).dot(u1) + a_mat1.dot(v2).dot(u2))

    a_matfail = np.random.rand(n, n, 3, 4) + mc * np.random.rand(n, n, 3, 4)
    pytest.raises(ValueError, a.contract, a_matfail)

    # Batch rows
    a_submat = np.random.rand(3, 4) + mc * np.random.rand(3, 4)
    rows = np.array([[3, 5, 5], [5, 6, 7], [8, 9, 0]])
    cols = np.array([0, 1, 2, 8])
    assert a.contract(a_submat, rows, cols).shape == (3,)
    npt.assert_allclose(a.contract(a_submat, rows, cols),
                        u1[rows].dot(a_submat.dot(v1[cols])) + u2[rows].dot(a_submat.dot(v2[cols])))

    assert a.contract(rows=rows, cols=rows).shape == (3,)
    npt.assert_allclose(a.contract(rows=rows, cols=rows),
                        np.sum(u1[rows] * v1[rows], axis=1) + np.sum(u2[rows] * v2[rows], axis=1))

    cols_fail = np.array([[0, 1, 2, 8], [3, 4, 7, 8]])
    pytest.raises(ValueError, a.contract, rows=rows, cols=cols_fail)

    # Batch cols
    a_submat1 = np.random.rand(3, 4) + mc * np.random.rand(3, 4)
    rows = np.array([3, 5, 5])
    cols = np.array([[0, 1, 2, 8], [3, 4, 7, 8], [5, 4, 1, 0]])
    assert a.contract(a_submat1, rows, cols).shape == (3,)
    npt.assert_allclose(a.contract(a_submat1, rows, cols),
                        u1[rows].dot(a_submat1.dot(v1[cols].T)) + u2[rows].dot(a_submat1.dot(v2[cols].T)))

    # Batch all
    rows = np.array([[3, 5, 5], [5, 6, 7], [8, 9, 0]])
    cols = np.array([[0, 1, 2, 8], [3, 4, 7, 8], [5, 4, 1, 0]])
    a_submat2 = np.random.rand(3, 3, 4) + mc * np.random.rand(3, 3, 4)
    npt.assert_allclose(a.contract(a_submat2, rows, cols)[0],
                        a.contract(a_submat2[0, :], rows[0, :], cols[0, :]))
    npt.assert_allclose(a.contract(a_submat2, rows, cols)[1],
                        a.contract(a_submat2[1, :], rows[1, :], cols[1, :]))
    npt.assert_allclose(a.contract(a_submat2, rows, cols)[2],
                        a.contract(a_submat2[2, :], rows[2, :], cols[2, :]))

    rows_fail = np.array([[3, 5, 5], [5, 6, 7]])
    pytest.raises(ValueError, a.contract, a_submat2, rows_fail, cols)


if __name__ == '__main__':
    pytest.main([__file__])
