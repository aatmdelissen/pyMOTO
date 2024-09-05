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
        """ Test one element in the middle without boundary effects """
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
        npt.assert_allclose(y[domain.get_elemnumber(ix+1, iy)], w[0,1,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy+1)], w[1,0,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix-1, iy)], w[2,1,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy-1)], w[1,2,0])
        npt.assert_allclose(y[domain.get_elemnumber(ix+1, iy+1)], w[0, 0, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix-1, iy+1)], w[2, 0, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix-1, iy-1)], w[2, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix+1, iy-1)], w[0, 2, 0])

    def test_2D_edge_xmin_symmetric(self):
        """ Test one element at the edge to test xmin boundary effect """
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12)

        ix, iy = 0, 6

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(sx, domain=domain, radius=2, relative_units=True, xmin_bc='symmetric')
        m.response()

        y = m.sig_out[0].state
        w = m.weights

        npt.assert_allclose(y[domain.get_elemnumber(ix, iy - 1)], w[1, 2, 0] + w[2, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy)],     w[1, 1, 0] + w[2, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy + 1)], w[1, 0, 0] + w[2, 0, 0])

        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy - 1)], w[0, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy)],     w[0, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy + 1)], w[0, 0, 0])

    def test_2D_edge_xmin_constval(self):
        """ Test one element at the edge to test xmin boundary effect """
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12)

        ix, iy = 0, 6

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(sx, domain=domain, radius=2, relative_units=True, xmin_bc=0.0)
        m.response()

        y = m.sig_out[0].state
        w = m.weights

        npt.assert_allclose(y[domain.get_elemnumber(ix, iy - 1)], w[1, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy)],     w[1, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy + 1)], w[1, 0, 0])

        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy - 1)], w[0, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy)],     w[0, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy + 1)], w[0, 0, 0])

    def test_2D_edge_xmax_symmetric(self):
        """ Test one element at the edge to test xmin boundary effect """
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12)

        ix, iy = domain.nelx-1, 6

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(sx, domain=domain, radius=2, relative_units=True, xmin_bc='symmetric')
        m.response()

        y = m.sig_out[0].state
        w = m.weights

        npt.assert_allclose(y[domain.get_elemnumber(ix, iy - 1)], w[1, 2, 0] + w[0, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy)],     w[1, 1, 0] + w[0, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy + 1)], w[1, 0, 0] + w[0, 0, 0])

        npt.assert_allclose(y[domain.get_elemnumber(ix - 1, iy - 1)], w[2, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix - 1, iy)],     w[2, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix - 1, iy + 1)], w[2, 0, 0])

    def test_2D_fd_symmetric_kernel(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12, unitx=0.5, unity=1.0)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))

        m = pym.FilterConv(sx, domain=domain, radius=5.3, relative_units=False)

        pym.finite_difference(m, test_fn=fd_testfn)

    def test_2D_fd_asymmetric_kernel(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 21)

        x = np.random.rand(domain.nel)

        sx = pym.Signal('x', state=x)
        nx, ny = 3, 1
        weights = 1+np.arange(nx*ny).reshape((nx, ny, 1))
        m = pym.FilterConv(sx, domain=domain, weights=weights)

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

    def test_3D_dot_symm_at_zmin(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 11, 12)

        ix, iy, iz = 5, 6, 0

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy, iz)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(sx, domain=domain, radius=2, relative_units=True)
        m.response()

        y = m.sig_out[0].state
        w = m.weights
        ysel = y[domain.elements[ix + np.arange(-1, 2), :, :][:, iy + np.arange(-1, 2), :][:, :, :2]]

        # Layer affected by symmetry
        npt.assert_allclose(ysel[0, 0, 0], w[0, 0, 1] + w[0, 0, 0])
        npt.assert_allclose(ysel[1, 0, 0], w[1, 0, 1] + w[1, 0, 0])
        npt.assert_allclose(ysel[2, 0, 0], w[2, 0, 1] + w[2, 0, 0])

        npt.assert_allclose(ysel[0, 1, 0], w[0, 1, 1] + w[0, 1, 0])
        npt.assert_allclose(ysel[1, 1, 0], w[1, 1, 1] + w[1, 1, 0])
        npt.assert_allclose(ysel[2, 1, 0], w[2, 1, 1] + w[2, 1, 0])

        npt.assert_allclose(ysel[0, 2, 0], w[0, 2, 1] + w[0, 2, 0])
        npt.assert_allclose(ysel[1, 2, 0], w[1, 2, 1] + w[1, 2, 0])
        npt.assert_allclose(ysel[2, 2, 0], w[2, 2, 1] + w[2, 2, 0])

        # Layer unaffected by symmetry
        npt.assert_allclose(ysel[0, 0, 1], w[0, 0, 2])
        npt.assert_allclose(ysel[1, 0, 1], w[1, 0, 2])
        npt.assert_allclose(ysel[2, 0, 1], w[2, 0, 2])

        npt.assert_allclose(ysel[0, 1, 1], w[0, 1, 2])
        npt.assert_allclose(ysel[1, 1, 1], w[1, 1, 2])
        npt.assert_allclose(ysel[2, 1, 1], w[2, 1, 2])

        npt.assert_allclose(ysel[0, 2, 1], w[0, 2, 2])
        npt.assert_allclose(ysel[1, 2, 1], w[1, 2, 2])
        npt.assert_allclose(ysel[2, 2, 1], w[2, 2, 2])

    def test_3D_symmetric(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12, 13, unitx=0.5, unity=1.0, unitz=1.2)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))

        m = pym.FilterConv(sx, domain=domain, radius=5.3, relative_units=False)

        pym.finite_difference(m, test_fn=fd_testfn)

    def test_3D_symmetric1(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(6, 6, 6, unitx=0.5, unity=1.0, unitz=1.2)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))
        start = time.time()
        m1 = pym.FilterConv(sx, domain=domain, radius=5, relative_units=True)
        print(f"Convolution setup = {time.time() - start} s")
        start = time.time()
        m1.response()
        print(f"Convolution elapsed = {time.time() - start} s")

        start = time.time()
        m2 = pym.DensityFilter(sx, domain=domain, radius=5)
        print(f"H-matrix setup = {time.time() - start} s")
        start = time.time()
        m2.response()
        print(f"H-matrix elapsed = {time.time() - start} s")

