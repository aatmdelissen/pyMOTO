import unittest
import numpy as np
import pymoto as pym
import numpy.testing as npt

class TestElMats(unittest.TestCase):
    def TestMassMat2D(self):
        N = 20
        Lx, Ly, Lz = 2, 2, 5
        lx, ly, lz = Lx/N, Ly/N, Lz
        domain = pym.DomainDefinition(N,N, unitx = lx, unity = ly, unitz = lz)
        rho = 1.0
        mel = rho*np.prod(domain.element_size)
        MEhc = mel / 36 * np.array([[4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                                      [0.0, 4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0],
                                      [2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                                      [0.0, 2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 2.0],
                                      [2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 2.0, 0.0],
                                      [0.0, 2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 2.0],
                                      [1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0, 0.0],
                                      [0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0]])
        ME = pym.ConsistentMassEq(domain, 2, 1.0)

        self.assertEqual(ME, MEhc)

    def TestConductivityMat(self):
        N = 20
        Lx, Ly, Lz = 2, 2, 5
        lx, ly, lz = Lx/N, Ly/N, Lz
        domain = pym.DomainDefinition(N,N, unitx = lx, unity = ly, unitz = lz)
        nodidx_left = domain.get_nodenumber(0, np.arange(domain.nely + 1))
        nodidx_right = domain.get_nodenumber(domain.nelx, np.arange(domain.nely + 1))

        kt = 1.0

        s_x = pym.Signal('x', state=np.ones(domain.nel))
        m_KT = pym.AssembleScalarField(s_x, domain=domain, bc=nodidx_left, kt = kt)
        s_KT = m_KT.sig_out[0]
        m_KT.response()

        q = np.zeros(domain.nnodes)
        Q = 1.0
        q[nodidx_right] = Q / nodidx_right.size

        # check with simple 1D heat conduction through wall
        T_chk = Q*Lx/(kt*Ly*Lz)
        T = np.linalg.solve(s_KT.state.toarray(), q)

        npt.assert_allclose(T[nodidx_right[0]], 0, atol=1e-10)
        npt.assert_allclose(T[nodidx_right[1]], T_chk, rtol=1e-10)


    def TestCapacitanceMat(self):




    def TestMassMat3D(self):


    def TestConductivityMat3D(self):
