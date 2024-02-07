import unittest
import numpy as np
import pymoto as pym
import numpy.testing as npt

np.random.seed(0)


class TestElementOperations(unittest.TestCase):
    def test_strain_xx(self):
        domain = pym.DomainDefinition(1, 1, unitx=0.7, unity=0.8)

        u = np.zeros(domain.elemnodes * domain.dim)
        n_right = domain.get_nodenumber(1, np.arange(2))

        idx_x = n_right * domain.dim
        disp = 0.1
        u[idx_x] = disp

        m_strain = pym.Strain(pym.Signal(state=u), domain=domain)
        m_strain.response()

        npt.assert_allclose(m_strain.sig_out[0].state[:, 0], np.array([disp/domain.unitx, 0, 0]))

        m_stress = pym.Stress(pym.Signal(state=u), domain=domain, e_modulus=67.0)
        m_stress.response()
        npt.assert_allclose(m_stress.sig_out[0].state[:, 0], np.array([67.0 / domain.unitx, 0, 0]))

