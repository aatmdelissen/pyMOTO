import unittest
import numpy as np
import pymoto as pym
import numpy.testing as npt


class TestAssembleStiffness(unittest.TestCase):
    def test_FEA_pure_tensile_2d_one_element(self):
        Lx, Ly, Lz = 0.1, 0.2, 0.3
        domain = pym.DomainDefinition(1, 1, unitx=Lx, unity=Ly, unitz=Lz)
        nodidx_left = domain.get_nodenumber(0, np.arange(domain.nely + 1))
        # Fixed at bottom, roller at the top in y-direction
        nodidx_right = domain.get_nodenumber(domain.nelx, np.arange(domain.nely + 1))
        dofidx_left = np.concatenate([nodidx_left*2, np.array([nodidx_left[0]*2 + 1])])

        E, nu = 210e+9, 0.3

        s_x = pym.Signal('x', state=np.ones(domain.nel))

        # Assemble stiffness matrix
        m_K = pym.AssembleStiffness(s_x, domain=domain, bc=dofidx_left, e_modulus=E, poisson_ratio=nu, plane='stress')
        s_K = m_K.sig_out[0]

        m_K.response()
        F = 1.5
        f = np.zeros(domain.nnodes*2)
        f[nodidx_right * 2] = F/nodidx_right.size
        x = np.linalg.solve(s_K.state.toarray(), f)

        # Bottom y displacement should be zero
        npt.assert_allclose(x[nodidx_right[0]*2+1], 0, atol=1e-10)

        # Analytical axial displacement using stiffness k = EA/L
        ux_chk = F * Lx / (E * Ly * Lz)
        npt.assert_allclose(x[nodidx_right*2], ux_chk, rtol=1e-10)

        # Transverse displacement using Poisson's effect
        e_xx = ux_chk / Lx  # Strain in x-direction
        e_yy = - nu * e_xx
        uy_chk = e_yy * Ly
        npt.assert_allclose(x[nodidx_right[1]*2+1], uy_chk, rtol=1e-10)

    def test_FEA_pure_tensile_3d_one_element(self):
        Lx, Ly, Lz = 0.1, 0.2, 0.3
        domain = pym.DomainDefinition(1, 1, 1, unitx=Lx, unity=Ly, unitz=Lz)
        nodidx_left = domain.get_nodenumber(*np.meshgrid(0, range(domain.nely + 1), range(domain.nelz + 1))).flatten()
        # Fixed at (0,0,0), roller in z-direction at (0, 1, 0), roller in y-direction at (0, 0, 1)
        nod_00 = domain.get_nodenumber(0, 0, 0)
        nod_10 = domain.get_nodenumber(0, 1, 0)
        nod_01 = domain.get_nodenumber(0, 0, 1)
        dofidx_left = np.concatenate([nodidx_left * 3, np.array([nod_00, nod_01]) * 3 + 1, np.array([nod_00, nod_10]) * 3 + 2])
        nodidx_right = domain.get_nodenumber(*np.meshgrid(1, range(domain.nely + 1), range(domain.nelz + 1))).flatten()

        E, nu = 210e+9, 0.3

        s_x = pym.Signal('x', state=np.ones(domain.nel))

        # Assemble stiffness matrix
        m_K = pym.AssembleStiffness(s_x, domain=domain, bc=dofidx_left, e_modulus=E, poisson_ratio=nu)
        s_K = m_K.sig_out[0]

        m_K.response()
        F = 1.5e+3
        f = np.zeros(domain.nnodes * 3)
        f[nodidx_right * 3] = F / nodidx_right.size
        x = np.linalg.solve(s_K.state.toarray(), f)

        # y and z displacements at (1, 0, 0) should be zero
        npt.assert_allclose(x[domain.get_nodenumber(1, 0, 0) * 3 + 1], 0, atol=1e-10)
        npt.assert_allclose(x[domain.get_nodenumber(1, 0, 0) * 3 + 2], 0, atol=1e-10)

        # Z displacement at (1, 1, 0) should be zero
        npt.assert_allclose(x[domain.get_nodenumber(1, 1, 0) * 3 + 2], 0, atol=1e-10)

        # Y displacement at (1, 0, 1) should be zero
        npt.assert_allclose(x[domain.get_nodenumber(1, 0, 1) * 3 + 1], 0, atol=1e-10)

        # Analytical axial displacement using stiffness k = EA/L
        ux_chk = F * Lx / (E * Ly * Lz)
        npt.assert_allclose(x[nodidx_right * 3], ux_chk, rtol=1e-10)

        # Transverse displacement using Poisson's effect
        e_xx = ux_chk / Lx  # Strain in x-direction
        e_trans = - nu * e_xx
        uy_chk = e_trans * Ly
        npt.assert_allclose(x[domain.get_nodenumber(1, 1, 0) * 3 + 1], uy_chk, rtol=1e-10)
        npt.assert_allclose(x[domain.get_nodenumber(1, 1, 1) * 3 + 1], uy_chk, rtol=1e-10)

        uz_chk = e_trans * Lz
        npt.assert_allclose(x[domain.get_nodenumber(1, 0, 1) * 3 + 2], uz_chk, rtol=1e-10)
        npt.assert_allclose(x[domain.get_nodenumber(1, 1, 1) * 3 + 2], uz_chk, rtol=1e-10)