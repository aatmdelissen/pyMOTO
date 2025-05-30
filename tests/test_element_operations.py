import pytest
import numpy as np
import pymoto as pym
import numpy.testing as npt


def fd_testfn(x0, dx, df_an, df_fd):
    npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)


class TestElementOperations:

    domains = dict(one_element_2D=pym.DomainDefinition(1, 1, unitx=0.7, unity=0.8),
                   multi_element_2D=pym.DomainDefinition(2, 2, unitx=0.7, unity=0.8),
                   one_element_3D=pym.DomainDefinition(1, 1, 1, unitx=0.7, unity=0.8, unitz=0.9),
                   multi_element_3D=pym.DomainDefinition(2, 2, 2, unitx=0.7, unity=0.8, unitz=0.9),
                   )

    @pytest.mark.parametrize('domain', domains.values(), ids=domains.keys())
    def test_0d_output(self, domain: pym.DomainDefinition):
        u = 0.1*np.arange(domain.nnodes * domain.dim)
        s_u = pym.Signal('u', state=u)

        dofconn = domain.get_dofconnectivity(domain.dim)

        # Average x displacement
        avg_ux = np.zeros((domain.dim*domain.elemnodes))
        avg_ux[::2] = 1/domain.elemnodes
        s_avgx = pym.ElementOperation(domain=domain, element_matrix=avg_ux)(s_u)

        npt.assert_allclose(s_avgx.state.shape, (domain.nel))
        ux_chk = np.sum(u[dofconn[:, ::2]], axis=-1)/domain.elemnodes
        npt.assert_allclose(s_avgx.state, ux_chk)

        pym.finite_difference(s_u, s_avgx, test_fn=fd_testfn)

        # Average y displacement
        avg_uy = np.zeros_like(avg_ux)
        avg_uy[1::2] = 1/domain.elemnodes
        s_avgy = pym.ElementOperation(domain=domain, element_matrix=avg_uy)(s_u)

        npt.assert_allclose(s_avgy.state.shape, (domain.nel))
        uy_chk = np.sum(u[dofconn[:, 1::2]], axis=-1)/domain.elemnodes
        npt.assert_allclose(s_avgy.state, uy_chk)

        pym.finite_difference(s_u, s_avgy, test_fn=fd_testfn)

    @pytest.mark.parametrize('domain', domains.values(), ids=domains.keys())
    def test_1d_output(self, domain: pym.DomainDefinition):
        u = 0.1*np.arange(domain.nnodes * domain.dim)
        s_u = pym.Signal('u', state=u)

        dofconn = domain.get_dofconnectivity(domain.dim)

        # Average (x, y) displacement
        avg_u = np.zeros((2, domain.dim*domain.elemnodes))
        avg_u[0, 0::2] = 1 / domain.elemnodes
        avg_u[1, 1::2] = 1 / domain.elemnodes
        s_avg = pym.ElementOperation(domain=domain, element_matrix=avg_u)(s_u)

        npt.assert_allclose(s_avg.state.shape, (2, domain.nel))
        ux_chk = np.sum(u[dofconn[:, ::2]], axis=-1)/domain.elemnodes
        uy_chk = np.sum(u[dofconn[:, 1::2]], axis=-1) / domain.elemnodes
        npt.assert_allclose(s_avg.state[0, :], ux_chk)
        npt.assert_allclose(s_avg.state[1, :], uy_chk)

        pym.finite_difference(test_fn=fd_testfn)

    @pytest.mark.parametrize('domain', domains.values(), ids=domains.keys())
    def test_2d_output(self, domain: pym.DomainDefinition):
        u = 0.1*np.arange(domain.nnodes * domain.dim)
        s_u = pym.Signal('u', state=u)

        dofconn = domain.get_dofconnectivity(domain.dim)

        # Average (x, y) displacement
        avg_u = np.zeros((2, 2, domain.dim*domain.elemnodes))
        avg_u[0, 0, 0::2] = 1 / domain.elemnodes
        avg_u[1, 1, 1::2] = 1 / domain.elemnodes
        s_avg = pym.ElementOperation(domain=domain, element_matrix=avg_u)(s_u)

        npt.assert_allclose(s_avg.state.shape, (2, 2, domain.nel))
        ux_chk = np.sum(u[dofconn[:, ::2]], axis=-1)/domain.elemnodes
        uy_chk = np.sum(u[dofconn[:, 1::2]], axis=-1) / domain.elemnodes
        npt.assert_allclose(s_avg.state[0, 0, :], ux_chk)
        npt.assert_allclose(s_avg.state[0, 1, :], 0)
        npt.assert_allclose(s_avg.state[1, 0, :], 0)
        npt.assert_allclose(s_avg.state[1, 1, :], uy_chk)

        pym.finite_difference(test_fn=fd_testfn)

    @pytest.mark.parametrize('domain', domains.values(), ids=domains.keys())
    def test_3d_output(self, domain: pym.DomainDefinition):
        u = 0.1 * np.arange(domain.nnodes * domain.dim)
        s_u = pym.Signal('u', state=u)

        dofconn = domain.get_dofconnectivity(domain.dim)

        # Displacement of each node per element
        avg_u = np.zeros((2, 2, 2, domain.dim * domain.elemnodes))
        for i, n in enumerate(domain.node_numbering):
            if i==4:
                break
            avg_u[(n[0]+1)//2, (n[1]+1)//2, 0, i*domain.dim+0] = 1.2
            avg_u[(n[0]+1)//2, (n[1]+1)//2, 1, i*domain.dim+1] = 1.1
        s_avg = pym.ElementOperation(domain=domain, element_matrix=avg_u)(s_u)

        npt.assert_allclose(s_avg.state.shape, (2, 2, 2, domain.nel))
        for i, n in enumerate(domain.node_numbering):
            if i == 4:
                break
            npt.assert_allclose(s_avg.state[(n[0]+1)//2, (n[1]+1)//2, 0, :], u[dofconn[:, i * domain.dim + 0]] * 1.2)
            npt.assert_allclose(s_avg.state[(n[0]+1)//2, (n[1]+1)//2, 1, :], u[dofconn[:, i * domain.dim + 1]] * 1.1)

        pym.finite_difference(test_fn=fd_testfn)

    def test_element_operation_repeat_multidim(self):
        np.random.seed(0)
        domain = pym.DomainDefinition(10, 11)
        em = np.array([[0.1, 0.2, 0.3, 0.4],  # EM defined on each node per element, not for each dof
                       [1.1, 1.2, 1.3, 1.4],
                       [2.1, 2.2, 2.3, 2.4]])

        x = pym.Signal(state=np.random.rand(2 * domain.nnodes))
        y = pym.ElementOperation(domain=domain, element_matrix=em)(x)

        y_chk1 = em @ x.state[domain.get_dofconnectivity(2)[:, ::2]].T
        y_chk2 = em @ x.state[domain.get_dofconnectivity(2)[:, 1::2]].T

        npt.assert_allclose(y.state[0], y_chk1)
        npt.assert_allclose(y.state[1], y_chk2)

        pym.finite_difference(test_fn=fd_testfn)


class TestStressStrain:
    def test_strain_xx(self):
        domain = pym.DomainDefinition(1, 1, unitx=0.6, unity=0.7)

        u = np.zeros(domain.elemnodes * domain.dim)
        n_right = domain.get_nodenumber(1, np.arange(2))

        idx_x = n_right * domain.dim
        disp = 0.1
        u[idx_x] = disp

        # Test strain
        s_strain = pym.Strain(domain=domain)(pym.Signal(state=u))
        exx_chk = disp/domain.unitx
        npt.assert_allclose(s_strain.state[:, 0], np.array([exx_chk, 0, 0]), atol=1e-16)
        pym.finite_difference(tosig=s_strain, test_fn=fd_testfn)

        # Test stress
        E = 67.0
        s_stress = pym.Stress(domain=domain, e_modulus=E, poisson_ratio=0.0)(pym.Signal(state=u))
        sxx_chk = E * exx_chk
        npt.assert_allclose(s_stress.state[:, 0], np.array([sxx_chk, 0, 0]), atol=1e-16)
        pym.finite_difference(tosig=s_stress, test_fn=fd_testfn)

    def test_pure_shear(self):
        domain = pym.DomainDefinition(1, 1, unitx=0.6, unity=0.7)

        u = np.zeros(domain.elemnodes * domain.dim)
        n_top = domain.get_nodenumber(np.arange(2), 1)

        idx_x = n_top * domain.dim
        disp = 0.1
        u[idx_x] = disp

        # Test strain
        s_strain = pym.Strain(domain=domain)(pym.Signal(state=u))
        gam_xy_chk = disp/domain.unity
        npt.assert_allclose(s_strain.state[:, 0], np.array([0, 0, 2*gam_xy_chk]), atol=1e-16)
        pym.finite_difference(tosig=s_strain, test_fn=fd_testfn)

        # Test stress
        E, nu = 67.0, 0.3
        s_stress = pym.Stress(domain=domain, e_modulus=E, poisson_ratio=nu)(pym.Signal(state=u))
        G = E / (2*(1+nu))
        sxy_chk = 2 * G * gam_xy_chk
        npt.assert_allclose(s_stress.state[:, 0], np.array([0, 0, sxy_chk]), atol=1e-16)
        pym.finite_difference(tosig=s_stress, test_fn=fd_testfn)
