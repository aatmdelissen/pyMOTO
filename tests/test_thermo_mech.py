import unittest
import numpy as np
import pymoto as pym
import numpy.testing as npt


class TestThermoMechanical(unittest.TestCase):
    def test_elemental_average(self):
        N = 10
        Lx, Ly, Lz = 1, 1, 1
        lx, ly, lz = Lx / N, Ly, Lz
        domain = pym.DomainDefinition(N, 1, unitx=lx, unity=ly, unitz=lz)

        T = pym.Signal("T", state=np.arange(domain.nnodes))
        m_avg = pym.ElementAverage(T, domain=domain)
        T_av = m_avg.sig_out[0]
        m_avg.response()

        start = 1.5 + 0.5*(N-1)
        T_avchk = np.arange(start, start+N)

        npt.assert_allclose(T_avchk, T_av.state)

    def test_thermal_expansion(self):
        Lx, Ly, Lz = 2, 1, 1
        domain = pym.DomainDefinition(10, 10, unitx=Lx/10, unity=Ly/10, unitz=Lz)

        # Fixed in the middle for free expansion
        nodidx_mid = domain.get_nodenumber(domain.nelx//2, np.arange(domain.nely//2, domain.nely//2 + 1))
        dofidx_mid = np.concatenate((nodidx_mid*2, nodidx_mid*2 + 1))

        # Define bottom, top, right, left surfaces
        nodidx_bottom = domain.get_nodenumber(domain.nelx//2, 0)
        nodidx_top = domain.get_nodenumber(domain.nelx//2, domain.nely)
        nodidx_right = domain.get_nodenumber(domain.nelx, domain.nely//2)
        nodidx_left = domain.get_nodenumber(0, domain.nely//2)

        E, nu, alpha = 100e+9, 0.0, 1e-5

        s_x = pym.Signal('x', state=np.ones(domain.nel))

        # Assemble stiffness matrix
        m_K = pym.AssembleStiffness(s_x, domain=domain, bc=dofidx_mid, e_modulus=E, poisson_ratio=nu, plane='strain')
        s_K = m_K.sig_out[0]
        m_K.response()

        # Determine equivalent thermal load assuming 1 degree temperature increase
        m_Fth = pym.ThermoMechanical(s_x, domain=domain, e_modulus=E, poisson_ratio=nu, alpha=alpha, plane='strain')
        s_Fth = m_Fth.sig_out[0]
        m_Fth.response()

        u = np.linalg.solve(s_K.state.toarray(), s_Fth.state)

        # Bottom y displacement should be -alpha*Ly/2
        npt.assert_allclose(u[2*nodidx_bottom + 1], -alpha * Ly / 2, atol=1e-10)

        # Top y displacement should be alpha*Ly/2
        npt.assert_allclose(u[2*nodidx_top + 1], alpha * Ly / 2, atol=1e-10)

        # Right x displacement should be alpha*Lx/2
        npt.assert_allclose(u[2*nodidx_right], alpha * Lx / 2, atol=1e-10)

        # Left x displacement should be -alpha*Lx/2
        npt.assert_allclose(u[2*nodidx_left], -alpha * Lx / 2, atol=1e-10)
