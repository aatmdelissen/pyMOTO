import pytest
import numpy as np
import pymoto as pym
import numpy.testing as npt
import time


def fd_testfn(x0, dx, df_an, df_fd, rtol=1e-2, atol=1e-5):
    npt.assert_allclose(df_an, df_fd, rtol=rtol, atol=atol)


class TestConvolutionFilter:
    def test_2d_dot(self):
        """ Test one element in the middle without boundary effects """
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12)

        ix, iy = 5, 6

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(domain=domain, radius=2, relative_units=True)
        sy = m(sx)

        y = sy.state
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

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_2d_edge_xmin_symmetric(self):
        """ Test one element at the edge to test xmin boundary effect """
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12)

        ix, iy = 0, 6

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(domain=domain, radius=2, relative_units=True, xmin_bc='symmetric')
        sy = m(sx)

        y = sy.state
        w = m.weights

        npt.assert_allclose(y[domain.get_elemnumber(ix, iy - 1)], w[1, 2, 0] + w[2, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy)],     w[1, 1, 0] + w[2, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy + 1)], w[1, 0, 0] + w[2, 0, 0])

        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy - 1)], w[0, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy)],     w[0, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy + 1)], w[0, 0, 0])

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_2d_edge_xmin_constval(self):
        """ Test one element at the edge to test xmin boundary effect """
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12)

        ix, iy = 0, 6

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy)] = 1.0
        sx = pym.Signal('x', state=x)


        m = pym.FilterConv(domain=domain, radius=2, relative_units=True, xmin_bc=0.0)
        sy = m(sx)

        y = sy.state
        w = m.weights

        npt.assert_allclose(y[domain.get_elemnumber(ix, iy - 1)], w[1, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy)],     w[1, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy + 1)], w[1, 0, 0])

        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy - 1)], w[0, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy)],     w[0, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix + 1, iy + 1)], w[0, 0, 0])

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_2d_edge_xmax_symmetric(self):
        """ Test one element at the edge to test xmin boundary effect """
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12)

        ix, iy = domain.nelx-1, 6

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(domain=domain, radius=2, relative_units=True, xmin_bc='symmetric')
        sy = m(sx)

        y = sy.state
        w = m.weights

        npt.assert_allclose(y[domain.get_elemnumber(ix, iy - 1)], w[1, 2, 0] + w[0, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy)],     w[1, 1, 0] + w[0, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix, iy + 1)], w[1, 0, 0] + w[0, 0, 0])

        npt.assert_allclose(y[domain.get_elemnumber(ix - 1, iy - 1)], w[2, 2, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix - 1, iy)],     w[2, 1, 0])
        npt.assert_allclose(y[domain.get_elemnumber(ix - 1, iy + 1)], w[2, 0, 0])

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_2d_fd_symmetric_kernel(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12, unitx=0.5, unity=1.0)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))

        sy = pym.FilterConv(domain=domain, radius=5.3, relative_units=False)(sx)

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_2d_fd_asymmetric_kernel(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 21)

        x = np.random.rand(domain.nel)

        sx = pym.Signal('x', state=x)
        nx, ny = 3, 1
        weights = 1+np.arange(nx*ny).reshape((nx, ny, 1))
        sy = pym.FilterConv(domain=domain, weights=weights)(sx)

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_3d_dot(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 11, 12)

        ix, iy, iz = 5, 6, 7

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy, iz)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(domain=domain, radius=2, relative_units=True)
        sy = m(sx)

        y = sy.state
        w = m.weights

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    npt.assert_allclose(y[domain.get_elemnumber(ix-1+i, iy-1+j, iz-1+k)], w[i, j, k])

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_3d_dot_symm_at_zmin(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 11, 12)

        ix, iy, iz = 5, 6, 0

        x = np.zeros(domain.nel)
        x[domain.get_elemnumber(ix, iy, iz)] = 1.0
        sx = pym.Signal('x', state=x)

        m = pym.FilterConv(domain=domain, radius=2, relative_units=True)
        sy = m(sx)

        y = sy.state
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

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_3d_symmetric(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 12, 13, unitx=0.5, unity=1.0, unitz=1.2)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))

        sy = pym.FilterConv(domain=domain, radius=5.3, relative_units=False)(sx)

        pym.finite_difference(sx, sy, test_fn=fd_testfn)

    def test_3d_symmetric1(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(100, 100, 100, unitx=0.5, unity=1.0, unitz=1.2)

        sx = pym.Signal('x', state=np.random.rand(domain.nel))
        start = time.time()
        m1 = pym.FilterConv(domain=domain, radius=5, relative_units=True)
        sy1 = m1(sx)
        print(f"Convolution setup = {time.time() - start} s")
        start = time.time()
        m1.response()
        print(f"Convolution elapsed = {time.time() - start} s")

        start = time.time()
        m2 = pym.DensityFilter(domain=domain, radius=5)
        sy2 = m2(sx)
        print(f"H-matrix setup = {time.time() - start} s")
        start = time.time()
        m2.response()
        print(f"H-matrix elapsed = {time.time() - start} s")

        # Not a perfect match, as BC is different
        npt.assert_allclose(abs(sy1.state - sy2.state).min(), 0.0, atol=1e-4)


class TestOverhangFilter:
    def test_cone_2d(self):
        n = 40
        n2 = int(n/2)
        domain = pym.DomainDefinition(n, n)#, n)

        x = np.zeros(domain.nel)
        if domain.dim == 2:
            x[domain.elements[1:-1,1:-1]] = 1
            x[domain.elements[n2, :]] = 1
            x[domain.elements[:, n2]] = 1
        else: 
            x[domain.elements[1:-1,1:-1, 1:-1]] = 1
            x[domain.elements[n2, n2, :]] = 1
            x[domain.elements[n2, :, n2]] = 1
            x[domain.elements[:, n2, n2]] = 1


        direction = '-x'
        angle = 25.0
        _y = pym.OverhangFilter(domain, direction, overhang_angle=angle)(x)
        # m = pym.PlotDomain(domain)
        # m(y)
        # plt.show(block=True)


    @pytest.mark.parametrize('direction', [[1, 0], [-1, 0], [0, 1], [-1, 0], [1, 0, 0], '+x', '-y'])
    @pytest.mark.parametrize('nx', [1, 4])
    @pytest.mark.parametrize('ny', [1, 4])
    def test_sensitivity_2D(self, direction, nx, ny):
        domain = pym.DomainDefinition(nx, ny)
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(domain.nel))

        sy = pym.OverhangFilter(domain, direction)(sx)

        pym.finite_difference(sx, sy, test_fn=fd_testfn, dx=1e-6)

    @pytest.mark.parametrize('direction', [[1, 0], [-1, 0], [0, 1], [-1, 0]])
    @pytest.mark.parametrize('overhang_angle', [10, 30, 45, 60])
    def test_overhangs_2D(self, direction, overhang_angle):
        domain = pym.DomainDefinition(6, 6)
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(domain.nel))

        sy = pym.OverhangFilter(domain, direction, overhang_angle=overhang_angle)(sx)

        pym.finite_difference(sx, sy, test_fn=fd_testfn, dx=1e-6)

    @pytest.mark.parametrize('direction', [[1, 0, 0], '-x', '+y', [0, -1, 0], [0, 0, 1], '-z'])
    @pytest.mark.parametrize('overhang_angle', [10, 45, 60])
    @pytest.mark.parametrize('nsampling', [4, 7])
    def test_sensitivity_3D(self, direction, overhang_angle, nsampling):
        nx, ny, nz = 4, 4, 4
        domain = pym.DomainDefinition(nx, ny, nz)
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(domain.nel))

        sy = pym.OverhangFilter(domain, direction, overhang_angle=overhang_angle, nsampling=nsampling)(sx)

        pym.finite_difference(sx, sy, test_fn=fd_testfn, dx=1e-6)

if __name__ == '__main__':
    pytest.main([__file__])
