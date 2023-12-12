import unittest
import numpy as np
import pymoto as pym

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




    def TestCapacitanceMat(self):




    def TestMassMat3D(self):


    def TestConductivityMat3D(self):
