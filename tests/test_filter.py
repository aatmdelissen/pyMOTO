import unittest
import numpy as np
import pymoto as pym
import numpy.testing as npt
import matplotlib.pyplot as plt
import time


def fd_testfn(x0, dx, df_an, df_fd, rtol=1e-5, atol=1e-5):
    npt.assert_allclose(df_an, df_fd, rtol=rtol, atol=atol)


class TestConvolutionFilter(unittest.TestCase):
    def test_2D_dot(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12)

        ix, iy = 5, 6

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(sx, domain=domain, radius=2, relative_units=True)
        m.response()

        y = m.sig_out[0].state
        w = m.weights

        npt.assert_allclose(y[domain.get_elemnumber(ix, iy)], w[1,1,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix+1, iy)], w[2,1,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy+1)], w[1,2,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix-1, iy)], w[0,1,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy-1)], w[1,0,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix+1, iy+1)], w[2, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix-1, iy+1)], w[0, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix-1, iy-1)], w[0, 0, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix+1, iy-1)], w[2, 0, 0])

    def test_2D_symmetric(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12, unitx=0.5, unity=1.0)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))

        m = pym.FilterConv(sx, domain=domain, radius=5.3, relative_units=False)

        pym.finite_difference(m, test_fn=fd_testfn)

    def test_2D_asymmetric(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 21)

        x = np.random.rand(domain.nel)

        sx = pym.Signal('x', state=x)
        nx, ny = 3, 1
        weights = 1+np.arange(nx*ny).reshape((nx, ny, 1))
        m = pym.FilterConv(sx, domain=domain, weights=weights, mode='reflect')

        pym.finite_difference(m, test_fn=fd_testfn)

    def test_3D_dot(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 11, 12)

        ix, iy, iz = 5, 6, 7

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy, iz)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(sx, domain=domain, radius=2, relative_units=True)
        m.response()

        y = m.sig_out[0].state
        w = m.weights

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    npt.assert_allclose(y[domain.get_elemnumber(ix-1+i, iy-1+j, iz-1+k)], w[i,j,k])

    def test_3D_symmetric(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12, 13, unitx=0.5, unity=1.0, unitz=1.2)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))

        m = pym.FilterConv(sx, domain=domain, radius=5.3, relative_units=False)

        pym.finite_difference(m, test_fn=fd_testfn)

    def test_3D_symmetric1(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(100, 120, 13, unitx=0.5, unity=1.0, unitz=1.2)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))
        start = time.time()
        m1 = pym.FilterConv(sx, domain=domain, radius=5.3, relative_units=True)
        print(f"Convolution setup = {time.time() - start} s")
        start = time.time()
        m1.response()
        print(f"Convolution elapsed = {time.time() - start} s")

        start = time.time()
        m2 = pym.DensityFilter(sx, domain=domain, radius=5.3)
        print(f"H-matrix setup = {time.time() - start} s")
        start = time.time()
        m2.response()
        print(f"H-matrix elapsed = {time.time() - start} s")

