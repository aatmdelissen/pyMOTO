import inspect
import pathlib  # For importing files
import sys
import unittest

import numpy as np
import numpy.testing as npt

import pymoto as pym


class TestMultigrid(unittest.TestCase):
    def test_interpolation_2D(self):
        domain = pym.DomainDefinition(10, 10)
        mg1 = pym.solvers.GeometricMultigrid(domain)
        bc_nodes = domain.nodes[:, 0].flatten()
        bc = np.concatenate([bc_nodes*2, bc_nodes*2+1])

        sx = pym.Signal('x', np.ones(domain.nel))
        m = pym.AssembleStiffness(sx, domain=domain, bc=bc)
        m.response()
        K = m.sig_out[0].state

        mg1.update(K)

        # Test restriction fine -> coarse
        uf = np.ones(domain.nnodes * 2)
        uc = mg1.R.T @ uf
        uc = uc.reshape(int(domain.nelx/2)+1, int(domain.nely/2)+1, 2)
        npt.assert_allclose(uc[1:-1, :, :][:, 1:-1, :], 4)
        npt.assert_allclose(uc[0, 1:-1, :], 3)
        npt.assert_allclose(uc[-1, 1:-1, :], 3)
        npt.assert_allclose(uc[1:-1, 0, :], 3)
        npt.assert_allclose(uc[1:-1, -1, :], 3)

        npt.assert_allclose(uc[0, 0, :], 2.25)
        npt.assert_allclose(uc[0, -1, :], 2.25)
        npt.assert_allclose(uc[-1, -1, :], 2.25)
        npt.assert_allclose(uc[-1, 0, :], 2.25)

        # Test interpolation coarse -> fine
        uc = np.ones(int(domain.nelx / 2 + 1) * int(domain.nely / 2 + 1) * 2)
        uf = mg1.R @ uc
        npt.assert_allclose(uf, 1.0)

    def test_interpolation_3D(self):
        domain = pym.DomainDefinition(10, 10, 10)
        mg1 = pym.solvers.GeometricMultigrid(domain)
        bc_nodes = domain.nodes[:, :, 0].flatten()
        bc = np.concatenate([bc_nodes * 3, bc_nodes * 3 + 1, bc_nodes * 3 + 2])

        sx = pym.Signal('x', np.ones(domain.nel))
        m = pym.AssembleStiffness(sx, domain=domain, bc=bc)
        m.response()
        K = m.sig_out[0].state

        mg1.update(K)

        # Test restriction fine -> coarse
        uf = np.ones(domain.nnodes * 3)
        uc = mg1.R.T @ uf
        uc = uc.reshape(int(domain.nelx/2)+1, int(domain.nely/2)+1, int(domain.nelz/2)+1, 3)
        npt.assert_allclose(uc[1:-1, :, :, :][:, 1:-1, :, :][:, :, 1:-1, :], 8)
        npt.assert_allclose(uc[0, 1:-1, :, :][:, 1:-1, :], 6)
        npt.assert_allclose(uc[-1, 1:-1, :, :][:, 1:-1, :], 6)
        npt.assert_allclose(uc[1:-1, 0, :, :][:, 1:-1, :], 6)
        npt.assert_allclose(uc[1:-1, -1, :, :][:, 1:-1, :], 6)
        npt.assert_allclose(uc[1:-1, :, 0, :][:, 1:-1, :], 6)
        npt.assert_allclose(uc[1:-1, :, -1, :][:, 1:-1, :], 6)

        npt.assert_allclose(uc[0, 0, 1:-1, :], 4.5)
        npt.assert_allclose(uc[0, -1, 1:-1, :], 4.5)
        npt.assert_allclose(uc[-1, -1, 1:-1, :], 4.5)
        npt.assert_allclose(uc[-1, 0, 1:-1, :], 4.5)
        npt.assert_allclose(uc[0, 1:-1, 0, :], 4.5)
        npt.assert_allclose(uc[0, 1:-1, -1, :], 4.5)
        npt.assert_allclose(uc[-1, 1:-1, -1, :], 4.5)
        npt.assert_allclose(uc[-1, 1:-1, 0, :], 4.5)
        npt.assert_allclose(uc[1:-1, 0, 0, :], 4.5)
        npt.assert_allclose(uc[1:-1, 0, -1, :], 4.5)
        npt.assert_allclose(uc[1:-1, -1, -1, :], 4.5)
        npt.assert_allclose(uc[1:-1, -1, 0, :], 4.5)

        npt.assert_allclose(uc[0, 0, 0, :], 3.375)
        npt.assert_allclose(uc[-1, 0, 0, :], 3.375)
        npt.assert_allclose(uc[0, -1, 0, :], 3.375)
        npt.assert_allclose(uc[-1, -1, 0, :], 3.375)
        npt.assert_allclose(uc[0, 0, -1, :], 3.375)
        npt.assert_allclose(uc[-1, 0, -1, :], 3.375)
        npt.assert_allclose(uc[0, -1, -1, :], 3.375)
        npt.assert_allclose(uc[-1, -1, -1, :], 3.375)

        # Test interpolation coarse -> fine
        uc = np.ones(int(domain.nelx / 2 + 1) * int(domain.nely / 2 + 1) * int(domain.nelz / 2 + 1) * 3)
        uf = mg1.R @ uc
        npt.assert_allclose(uf, 1.0)
