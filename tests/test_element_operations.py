import unittest
import numpy as np
import pymoto as pym
import numpy.testing as npt

np.random.seed(0)

def fd_testfn(x0, dx, df_an, df_fd):
    npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)

class TestElementOperations(unittest.TestCase):
    def test_0D_output(self):
        domains = dict()
        domains.update(one_element_2D=pym.DomainDefinition(1, 1, unitx=0.7, unity=0.8))
        domains.update(multi_element_2D=pym.DomainDefinition(2, 2, unitx=0.7, unity=0.8))
        domains.update(one_element_3D=pym.DomainDefinition(1, 1, 1, unitx=0.7, unity=0.8, unitz=0.9))
        domains.update(multi_element_3D=pym.DomainDefinition(2, 2, 2, unitx=0.7, unity=0.8, unitz=0.9))
        for k, domain in domains.items():
            with self.subTest(k):
                u = 0.1*np.arange(domain.nnodes * domain.dim)
                s_u = pym.Signal('u', state=u)

                dofconn = domain.get_dofconnectivity(domain.dim)

                # Average x displacement
                avg_ux = np.zeros((domain.dim*domain.elemnodes))
                avg_ux[::2] = 1/domain.elemnodes
                m_avgx = pym.ElementOperation(s_u, domain=domain, element_matrix=avg_ux)
                m_avgx.response()

                npt.assert_allclose(m_avgx.sig_out[0].state.shape, (domain.nel))
                ux_chk = np.sum(u[dofconn[:, ::2]], axis=-1)/domain.elemnodes
                npt.assert_allclose(m_avgx.sig_out[0].state, ux_chk)

                pym.finite_difference(m_avgx, test_fn=fd_testfn)

                # Average y displacement
                avg_uy = np.zeros_like(avg_ux)
                avg_uy[1::2] = 1/domain.elemnodes
                m_avgy = pym.ElementOperation(s_u, domain=domain, element_matrix=avg_uy)
                m_avgy.response()

                npt.assert_allclose(m_avgy.sig_out[0].state.shape, (domain.nel))
                uy_chk = np.sum(u[dofconn[:, 1::2]], axis=-1)/domain.elemnodes
                npt.assert_allclose(m_avgy.sig_out[0].state, uy_chk)

                pym.finite_difference(m_avgy, test_fn=fd_testfn)

    def test_1D_output(self):
        domains = dict()
        domains.update(one_element_2D=pym.DomainDefinition(1, 1, unitx=0.7, unity=0.8))
        domains.update(multi_element_2D=pym.DomainDefinition(2, 2, unitx=0.7, unity=0.8))
        domains.update(one_element_3D=pym.DomainDefinition(1, 1, 1, unitx=0.7, unity=0.8, unitz=0.9))
        domains.update(multi_element_3D=pym.DomainDefinition(2, 2, 2, unitx=0.7, unity=0.8, unitz=0.9))
        for k, domain in domains.items():
            with self.subTest(k):
                u = 0.1*np.arange(domain.nnodes * domain.dim)
                s_u = pym.Signal('u', state=u)

                dofconn = domain.get_dofconnectivity(domain.dim)

                # Average (x, y) displacement
                avg_u = np.zeros((2, domain.dim*domain.elemnodes))
                avg_u[0, 0::2] = 1 / domain.elemnodes
                avg_u[1, 1::2] = 1 / domain.elemnodes
                m_avg = pym.ElementOperation(s_u, domain=domain, element_matrix=avg_u)
                m_avg.response()

                npt.assert_allclose(m_avg.sig_out[0].state.shape, (2, domain.nel))
                ux_chk = np.sum(u[dofconn[:, ::2]], axis=-1)/domain.elemnodes
                uy_chk = np.sum(u[dofconn[:, 1::2]], axis=-1) / domain.elemnodes
                npt.assert_allclose(m_avg.sig_out[0].state[0, :], ux_chk)
                npt.assert_allclose(m_avg.sig_out[0].state[1, :], uy_chk)

                pym.finite_difference(m_avg, test_fn=fd_testfn)

    def test_2D_output(self):
        domains = dict()
        domains.update(one_element_2D=pym.DomainDefinition(1, 1, unitx=0.7, unity=0.8))
        domains.update(multi_element_2D=pym.DomainDefinition(2, 2, unitx=0.7, unity=0.8))
        domains.update(one_element_3D=pym.DomainDefinition(1, 1, 1, unitx=0.7, unity=0.8, unitz=0.9))
        domains.update(multi_element_3D=pym.DomainDefinition(2, 2, 2, unitx=0.7, unity=0.8, unitz=0.9))
        for k, domain in domains.items():
            with self.subTest(k):
                u = 0.1*np.arange(domain.nnodes * domain.dim)
                s_u = pym.Signal('u', state=u)

                dofconn = domain.get_dofconnectivity(domain.dim)

                # Average (x, y) displacement
                avg_u = np.zeros((2, 2, domain.dim*domain.elemnodes))
                avg_u[0, 0, 0::2] = 1 / domain.elemnodes
                avg_u[1, 1, 1::2] = 1 / domain.elemnodes
                m_avg = pym.ElementOperation(s_u, domain=domain, element_matrix=avg_u)
                m_avg.response()

                npt.assert_allclose(m_avg.sig_out[0].state.shape, (2, 2, domain.nel))
                ux_chk = np.sum(u[dofconn[:, ::2]], axis=-1)/domain.elemnodes
                uy_chk = np.sum(u[dofconn[:, 1::2]], axis=-1) / domain.elemnodes
                npt.assert_allclose(m_avg.sig_out[0].state[0, 0, :], ux_chk)
                npt.assert_allclose(m_avg.sig_out[0].state[0, 1, :], 0)
                npt.assert_allclose(m_avg.sig_out[0].state[1, 0, :], 0)
                npt.assert_allclose(m_avg.sig_out[0].state[1, 1, :], uy_chk)

                pym.finite_difference(m_avg, test_fn=fd_testfn)

    def test_3D_output(self):
        domains = dict()
        domains.update(one_element_2D=pym.DomainDefinition(1, 1, unitx=0.7, unity=0.8))
        domains.update(multi_element_2D=pym.DomainDefinition(2, 2, unitx=0.7, unity=0.8))
        domains.update(one_element_3D=pym.DomainDefinition(1, 1, 1, unitx=0.7, unity=0.8, unitz=0.9))
        domains.update(multi_element_3D=pym.DomainDefinition(2, 2, 2, unitx=0.7, unity=0.8, unitz=0.9))
        for k, domain in domains.items():
            with self.subTest(k):
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
                m_avg = pym.ElementOperation(s_u, domain=domain, element_matrix=avg_u)
                m_avg.response()

                npt.assert_allclose(m_avg.sig_out[0].state.shape, (2, 2, 2, domain.nel))
                for i, n in enumerate(domain.node_numbering):
                    if i==4:
                        break
                    npt.assert_allclose(m_avg.sig_out[0].state[(n[0]+1)//2, (n[1]+1)//2, 0, :], u[dofconn[:, i * domain.dim + 0]] * 1.2)
                    npt.assert_allclose(m_avg.sig_out[0].state[(n[0]+1)//2, (n[1]+1)//2, 1, :], u[dofconn[:, i * domain.dim + 1]] * 1.1)

                pym.finite_difference(m_avg, test_fn=fd_testfn)


class TestStressStrain(unittest.TestCase):
    def test_strain_xx(self):
        domain = pym.DomainDefinition(1, 1, unitx=0.6, unity=0.7)

        u = np.zeros(domain.elemnodes * domain.dim)
        n_right = domain.get_nodenumber(1, np.arange(2))

        idx_x = n_right * domain.dim
        disp = 0.1
        u[idx_x] = disp

        m_strain = pym.Strain(pym.Signal(state=u), domain=domain)
        m_strain.response()

        exx_chk = disp/domain.unitx
        npt.assert_allclose(m_strain.sig_out[0].state[:, 0], np.array([exx_chk, 0, 0]), atol=1e-16)

        E = 67.0
        m_stress = pym.Stress(pym.Signal(state=u), domain=domain, e_modulus=E, poisson_ratio=0.0)
        m_stress.response()
        sxx_chk = E * exx_chk
        npt.assert_allclose(m_stress.sig_out[0].state[:, 0], np.array([sxx_chk, 0, 0]), atol=1e-16)

    def test_pure_shear(self):
        domain = pym.DomainDefinition(1, 1, unitx=0.6, unity=0.7)

        u = np.zeros(domain.elemnodes * domain.dim)
        n_top = domain.get_nodenumber(np.arange(2), 1)

        idx_x = n_top * domain.dim
        disp = 0.1
        u[idx_x] = disp

        m_strain = pym.Strain(pym.Signal(state=u), domain=domain)
        m_strain.response()

        gam_xy_chk = disp/domain.unity
        npt.assert_allclose(m_strain.sig_out[0].state[:, 0], np.array([0, 0, 2*gam_xy_chk]), atol=1e-16)

        E, nu = 67.0, 0.3
        m_stress = pym.Stress(pym.Signal(state=u), domain=domain, e_modulus=E, poisson_ratio=nu)
        m_stress.response()

        G = E / (2*(1+nu))
        sxy_chk = 2 * G * gam_xy_chk
        npt.assert_allclose(m_stress.sig_out[0].state[:, 0], np.array([0, 0, sxy_chk]), atol=1e-16)
