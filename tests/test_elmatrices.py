import unittest
import numpy as np
import pymoto as pym
import numpy.testing as npt


class TestElMats(unittest.TestCase):
    def test_mass_mat_2d(self):
        N = 1
        Lx, Ly, Lz = 2, 3, 4
        lx, ly, lz = Lx/N, Ly/N, Lz
        domain = pym.DomainDefinition(N, N, unitx=lx, unity=ly, unitz=lz)
        rho = 1.0

        # Hard coded mass element matrix, taken from Cook (eq 11.3-6)
        mel = rho*np.prod(domain.element_size)
        MEhc = mel / 36 * np.array([[4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                                    [0.0, 4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0],
                                    [2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                                    [0.0, 2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 2.0],
                                    [2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 2.0, 0.0],
                                    [0.0, 2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 2.0],
                                    [1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0, 0.0],
                                    [0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0]])

        s_x = pym.Signal('x', state=np.ones(domain.nel))
        m_M = pym.AssembleMass(s_x, domain=domain, material_property=rho, ndof=domain.dim)
        ME = m_M.el_mat

        npt.assert_allclose(ME, MEhc)

    def test_conductivity_mat_2d(self):
        N = 1
        Lx, Ly, Lz = 2, 3, 4
        lx, ly, lz = Lx/N, Ly/N, Lz
        domain = pym.DomainDefinition(N, N, unitx=lx, unity=ly, unitz=lz)
        nodidx_left = domain.get_nodenumber(0, np.arange(domain.nely + 1))
        nodidx_right = domain.get_nodenumber(domain.nelx, np.arange(domain.nely + 1))
        kt = 1.0

        s_x = pym.Signal('x', state=np.ones(domain.nel))
        m_KT = pym.AssemblePoisson(s_x, domain=domain, bc=nodidx_left, material_property=kt)
        s_KT = m_KT.sig_out[0]
        m_KT.response()

        q = np.zeros(domain.nnodes)
        Q = 1.0
        q[nodidx_right] = Q / nodidx_right.size

        # check with simple 1D heat conduction through wall Q = -k (dT/dx)
        T_chk = Q*Lx/(kt*Ly*Lz)
        T = np.linalg.solve(s_KT.state.toarray(), q)

        # Check if left boundary has T=0 and right boundary has T=T_chk
        npt.assert_allclose(T[nodidx_left], 0, atol=1e-10)
        npt.assert_allclose(T[nodidx_right], T_chk, rtol=1e-10)

    def test_capacitance_mat(self):
        N = 1
        Lx, Ly, Lz = 2, 3, 4
        lx, ly, lz = Lx/N, Ly/N, Lz
        domain = pym.DomainDefinition(N, N, unitx=lx, unity=ly, unitz=lz)
        rho = 1.0
        cp = 1.0

        # Hard coded thermal capacity matrix, taken from Cook eq 12.2-4, where it is a 1 dof version of mass element
        cel = cp*rho*np.prod(domain.element_size)
        CEhc = cel / 36 * np.array([[4.0, 2.0, 2.0, 1.0],
                                    [2.0, 4.0, 1.0, 2.0],
                                    [2.0, 1.0, 4.0, 2.0],
                                    [1.0, 2.0, 2.0, 4.0]])

        s_x = pym.Signal('x', state=np.ones(domain.nel))
        m_C = pym.AssembleMass(s_x, domain=domain, material_property=rho*cp, ndof=1)
        CE = m_C.el_mat

        npt.assert_allclose(CE, CEhc)

    def test_mass_mat_3d(self):
        N = 1
        Lx, Ly, Lz = 2, 3, 4
        lx, ly, lz = Lx/N, Ly/N, Lz/N
        domain = pym.DomainDefinition(N, N, N, unitx=lx, unity=ly, unitz=lz)
        rho = 1.0
        mel = rho * np.prod(domain.element_size)

        # Hard coded mass element matrix
        MEhc = np.zeros((domain.elemnodes * domain.dim, domain.elemnodes * domain.dim))
        weights = np.array([8.0, 4.0, 2.0, 1.0])
        for n1 in range(domain.elemnodes):
            for n2 in range(domain.elemnodes):
                dist = round(np.sum(abs(np.array(domain.node_numbering[n1]) - np.array(domain.node_numbering[n2]))) / 2)
                MEhc[n1 * domain.dim + np.arange(domain.dim), n2 * domain.dim + np.arange(domain.dim)] = weights[dist]
        MEhc *= mel / 216

        s_x = pym.Signal('x', state=np.ones(domain.nel))
        m_M = pym.AssembleMass(s_x, domain=domain, material_property=rho, ndof=domain.dim)
        ME = m_M.el_mat

        npt.assert_allclose(ME, MEhc)

    def test_conductivity_mat_3d(self):
        N = 1
        Lx, Ly, Lz = 2, 3, 4
        lx, ly, lz = Lx / N, Ly / N, Lz / N
        domain = pym.DomainDefinition(N, N, N, unitx=lx, unity=ly, unitz=lz)
        nodidx_left = domain.get_nodenumber(*np.meshgrid(0, range(domain.nely + 1), range(domain.nelz + 1))).flatten()
        nodidx_right = domain.get_nodenumber(*np.meshgrid(domain.nelx, range(domain.nely + 1), range(domain.nelz + 1))).flatten()
        kt = 1.0

        s_x = pym.Signal('x', state=np.ones(domain.nel))
        m_KT = pym.AssemblePoisson(s_x, domain=domain, bc=nodidx_left, material_property=kt)
        s_KT = m_KT.sig_out[0]
        m_KT.response()

        q = np.zeros(domain.nnodes)
        Q = 1.0
        q[nodidx_right] = Q / nodidx_right.size

        # check with simple 1D heat conduction through wall
        T_chk = Q * Lx / (kt * Ly * Lz)
        T = np.linalg.solve(s_KT.state.toarray(), q)

        # Check if left boundary has T=0 and right boundary has T=T_chk
        npt.assert_allclose(T[nodidx_left], 0, atol=1e-10)
        npt.assert_allclose(T[nodidx_right], T_chk, rtol=1e-10)