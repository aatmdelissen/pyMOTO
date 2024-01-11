import unittest
import numpy as np
import numpy.testing as npt
import pymoto as pym


class TestDomainDefinition(unittest.TestCase):
    def test_node_numbering_2D(self):
        Nx, Ny = 100, 142
        domain = pym.DomainDefinition(Nx, Ny, unitx=0.1, unity=0.2)
        i_nod, j_nod = 10, 20
        n_idx = domain.get_nodenumber(i_nod, j_nod)
        i_chk, j_chk = domain.get_node_indices(n_idx)
        self.assertEqual(i_nod, i_chk)
        self.assertEqual(j_nod, j_chk)
        i_pos, j_pos = domain.get_node_position(n_idx)
        npt.assert_allclose(i_pos, i_nod*0.1)
        npt.assert_allclose(j_pos, j_nod*0.2)

    def test_node_numbering_array_2D(self):
        Nx, Ny = 100, 142
        domain = pym.DomainDefinition(Nx, Ny, unitx=0.1, unity=0.2)
        i_nod, j_nod = np.meshgrid(np.arange(Nx+1), np.arange(Ny+1), indexing='ij')
        n_idx = domain.get_nodenumber(i_nod, j_nod)
        i_chk, j_chk = domain.get_node_indices(n_idx)
        npt.assert_equal(i_nod, i_chk)
        npt.assert_equal(j_nod, j_chk)
        self.assertEqual(np.min(n_idx), 0)  # Index must be between 0 and #nodes
        self.assertEqual(np.max(n_idx), domain.nnodes-1)
        self.assertEqual(n_idx.size, np.unique(n_idx.flatten()).size)  # All indices must be unique
        i_pos, j_pos = domain.get_node_position(n_idx)
        npt.assert_allclose(i_pos, i_nod * 0.1)
        npt.assert_allclose(j_pos, j_nod * 0.2)

    def test_node_numbering_3D(self):
        Nx, Ny, Nz = 100, 142, 284
        domain = pym.DomainDefinition(Nx, Ny, Nz, unitx=0.1, unity=0.2, unitz=0.3)
        i_nod, j_nod, k_nod = 10, 20, 30
        n_idx = domain.get_nodenumber(i_nod, j_nod, k_nod)
        i_chk, j_chk, k_chk = domain.get_node_indices(n_idx)
        self.assertEqual(i_nod, i_chk)
        self.assertEqual(j_nod, j_chk)
        self.assertEqual(k_nod, k_chk)
        i_pos, j_pos, k_pos = domain.get_node_position(n_idx)
        npt.assert_allclose(i_pos, i_nod * 0.1)
        npt.assert_allclose(j_pos, j_nod * 0.2)
        npt.assert_allclose(k_pos, k_nod * 0.3)

    def test_node_numbering_array_3D(self):
        Nx, Ny, Nz = 100, 142, 284
        domain = pym.DomainDefinition(Nx, Ny, Nz, unitx=0.1, unity=0.2, unitz=0.3)
        i_nod, j_nod, k_nod = np.meshgrid(np.arange(Nx+1), np.arange(Ny+1), np.arange(Nz+1), indexing='ij')
        n_idx = domain.get_nodenumber(i_nod, j_nod, k_nod)
        i_chk, j_chk, k_chk = domain.get_node_indices(n_idx)
        npt.assert_equal(i_nod, i_chk)
        npt.assert_equal(j_nod, j_chk)
        self.assertEqual(np.min(n_idx), 0)  # Index must be between 0 and #nodes
        self.assertEqual(np.max(n_idx), domain.nnodes-1)
        self.assertEqual(n_idx.size, np.unique(n_idx.flatten()).size)  # All indices must be unique
        i_pos, j_pos, k_pos = domain.get_node_position(n_idx)
        npt.assert_allclose(i_pos, i_nod * 0.1)
        npt.assert_allclose(j_pos, j_nod * 0.2)
        npt.assert_allclose(k_pos, k_nod * 0.3)


if __name__ == '__main__':
    unittest.main()
