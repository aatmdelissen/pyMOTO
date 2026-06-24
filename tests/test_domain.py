import pytest
import numpy as np
import numpy.testing as npt
import pymoto as pym
import matplotlib.pyplot as plt


def fd_testfn(x0, dx, df_an, df_fd):
    npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)


def test_node_numbering_2D():
    Nx, Ny = 100, 142
    domain = pym.VoxelDomain(Nx, Ny, unitx=0.1, unity=0.2)
    i_nod, j_nod = 10, 20
    n_idx = domain.get_nodenumber(i_nod, j_nod)
    i_chk, j_chk = domain.get_node_indices(n_idx)
    assert i_nod == i_chk
    assert j_nod == j_chk
    i_pos, j_pos = domain.get_node_position(n_idx)
    npt.assert_allclose(i_pos, i_nod*0.1)
    npt.assert_allclose(j_pos, j_nod*0.2)


@pytest.mark.parametrize("select_nodes_i", [0, slice(0, 10), np.array([0, 1, 2])])
@pytest.mark.parametrize("select_nodes_j", [0, slice(0, 10), np.array([0, 1, 2])])
@pytest.mark.parametrize("select_dofs", [0, np.array([0, 1])])
def test_dof_numbering_2D_2dofs(select_nodes_i, select_nodes_j, select_dofs):
    Nx, Ny = 100, 142
    domain = pym.VoxelDomain(Nx, Ny, unitx=0.1, unity=0.2)

    n_idx = domain.nodes[select_nodes_i, select_nodes_j]
    ndof = 2
    dof_idx = domain.get_dofnumber(n_idx, select_dofs, ndof)
    assert dof_idx.shape == (*np.shape(n_idx), *np.shape(select_dofs))

    def to_1d_array(x):
        return np.asarray(x).flatten()
    assert len(set(to_1d_array(dof_idx)//ndof) - set(to_1d_array(n_idx))) == 0  # All nodes are included at least once
    assert len(set(to_1d_array(dof_idx)%ndof) - set(to_1d_array(select_dofs))) == 0  # All dofs are included once


def test_node_numbering_array_2D():
    Nx, Ny = 100, 142
    domain = pym.VoxelDomain(Nx, Ny, unitx=0.1, unity=0.2)
    i_nod, j_nod = np.meshgrid(np.arange(Nx+1), np.arange(Ny+1), indexing='ij')
    n_idx = domain.get_nodenumber(i_nod, j_nod)
    i_chk, j_chk = domain.get_node_indices(n_idx)
    npt.assert_equal(i_nod, i_chk)
    npt.assert_equal(j_nod, j_chk)
    assert np.min(n_idx) == 0  # Index must be between 0 and #nodes
    assert np.max(n_idx) == domain.nnodes-1
    assert n_idx.size == np.unique(n_idx.flatten()).size  # All indices must be unique
    i_pos, j_pos = domain.get_node_position(n_idx)
    npt.assert_allclose(i_pos, i_nod * 0.1)
    npt.assert_allclose(j_pos, j_nod * 0.2)

def test_node_numbering_3D():
    Nx, Ny, Nz = 100, 142, 284
    domain = pym.VoxelDomain(Nx, Ny, Nz, unitx=0.1, unity=0.2, unitz=0.3)
    i_nod, j_nod, k_nod = 10, 20, 30
    n_idx = domain.get_nodenumber(i_nod, j_nod, k_nod)
    i_chk, j_chk, k_chk = domain.get_node_indices(n_idx)
    assert i_nod == i_chk
    assert j_nod == j_chk
    assert k_nod == k_chk
    i_pos, j_pos, k_pos = domain.get_node_position(n_idx)
    npt.assert_allclose(i_pos, i_nod * 0.1)
    npt.assert_allclose(j_pos, j_nod * 0.2)
    npt.assert_allclose(k_pos, k_nod * 0.3)

def test_node_numbering_array_3D():
    Nx, Ny, Nz = 100, 142, 284
    domain = pym.VoxelDomain(Nx, Ny, Nz, unitx=0.1, unity=0.2, unitz=0.3)
    i_nod, j_nod, k_nod = np.meshgrid(np.arange(Nx+1), np.arange(Ny+1), np.arange(Nz+1), indexing='ij')
    n_idx = domain.get_nodenumber(i_nod, j_nod, k_nod)
    i_chk, j_chk, k_chk = domain.get_node_indices(n_idx)
    npt.assert_equal(i_nod, i_chk)
    npt.assert_equal(j_nod, j_chk)
    assert np.min(n_idx) == 0  # Index must be between 0 and #nodes
    assert np.max(n_idx) == domain.nnodes-1
    assert n_idx.size == np.unique(n_idx.flatten()).size  # All indices must be unique
    i_pos, j_pos, k_pos = domain.get_node_position(n_idx)
    npt.assert_allclose(i_pos, i_nod * 0.1)
    npt.assert_allclose(j_pos, j_nod * 0.2)
    npt.assert_allclose(k_pos, k_nod * 0.3)

def test_shape_fn_2D():
    unitx, unity = 0.8, 0.3
    domain = pym.VoxelDomain(1, 1, unitx=unitx, unity=unity)

    for i, n in enumerate(domain.node_numbering):
        N_chk = np.zeros(domain.elemnodes)
        N_chk[i] = 1.0
        pos = np.array([n[0]*unitx/2, n[1]*unity/2])
        npt.assert_allclose(domain.eval_shape_fun(pos), N_chk)

def test_shape_fn_derivatives_2D():
    unitx, unity = 0.8, 0.3
    domain = pym.VoxelDomain(1, 1, unitx=unitx, unity=unity)
    pos = np.array([0.2, 0.1])

    class ShapeFn(pym.Module):
        def __call__(self, pos):
            return domain.eval_shape_fun(pos)

        def _sensitivity(self, dN):
            return domain.eval_shape_fun_der(pos) @ dN

    sp = pym.Signal('pos', state=pos)
    sn = ShapeFn()(sp)
    sn.tag = "N"
    pym.finite_difference(sp, sn, test_fn=fd_testfn)


def test_rigid_body_modes_full_2D():
    domain = pym.VoxelDomain(10, 10, unitx=0.1, unity=0.1)
    rbm = domain.get_rigid_body_modes()
    assert rbm.shape == (domain.nnodes*2, 3)
    assert np.all(rbm[::2, 0] == 1)
    assert np.all(rbm[1::2, 0] == 0)
    assert np.all(rbm[::2, 1] == 0)
    assert np.all(rbm[1::2, 1] == 1)
    com = np.array([0.5, 0.5])
    coords = domain.get_node_position()
    Rz = np.array([[0, -1], [1, 0]])
    rbm_rz = (Rz @ (coords - com[:, None])).T.flatten()
    assert np.allclose(rbm[:, 2], rbm_rz)


def test_rigid_body_modes_full_3D():
    domain = pym.VoxelDomain(10, 10, 10, unitx=0.1, unity=0.1, unitz=0.1)
    com = np.array([0.1, 0.2, 0.3])
    rbm = domain.get_rigid_body_modes(cor=com)
    assert rbm.shape == (domain.nnodes*3, 6)
    assert np.all(rbm[::3, 0] == 1)
    assert np.all(rbm[1::3, 0] == 0)
    assert np.all(rbm[2::3, 0] == 0)
    assert np.all(rbm[::3, 1] == 0)
    assert np.all(rbm[1::3, 1] == 1)
    assert np.all(rbm[2::3, 1] == 0)
    assert np.all(rbm[::3, 2] == 0)
    assert np.all(rbm[1::3, 2] == 0)
    assert np.all(rbm[2::3, 2] == 1)
    Rx, Ry, Rz = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    Rx[1, 2], Rx[2, 1] = -1, 1
    Ry[0, 2], Ry[2, 0] = 1, -1
    Rz[0, 1], Rz[1, 0] = -1, 1
    coords = domain.get_node_position()
    rbm_rx = (Rx @ (coords - com[:, None])).T.flatten()
    rbm_ry = (Ry @ (coords - com[:, None])).T.flatten()
    rbm_rz = (Rz @ (coords - com[:, None])).T.flatten()
    assert np.allclose(rbm[:, 3], rbm_rx)
    assert np.allclose(rbm[:, 4], rbm_ry)
    assert np.allclose(rbm[:, 5], rbm_rz)
    # domain.write_to_vti(dict(rbm=rbm))

def test_rigid_body_modes_subset_3D():
    domain = pym.VoxelDomain(10, 10, 10, unitx=0.1, unity=0.1, unitz=0.1)
    nod_idx = np.array([0, 1, 2, 3])
    com = np.average(domain.get_node_position(nod_idx), axis=1)
    rbm = domain.get_rigid_body_modes(nod_idx=nod_idx)
    assert rbm.shape == (domain.nnodes*3, 6)
    assert np.all(rbm[max(nod_idx + 1) * 3:, :] == 0)
    assert np.all(rbm[nod_idx*3+0, 0] == 1)
    assert np.all(rbm[nod_idx*3+1, 0] == 0)
    assert np.all(rbm[nod_idx*3+2, 0] == 0)
    assert np.all(rbm[nod_idx*3+0, 1] == 0)
    assert np.all(rbm[nod_idx*3+1, 1] == 1)
    assert np.all(rbm[nod_idx*3+2, 1] == 0)
    assert np.all(rbm[nod_idx*3+0, 2] == 0)
    assert np.all(rbm[nod_idx*3+1, 2] == 0)
    assert np.all(rbm[nod_idx*3+2, 2] == 1)
    Rx, Ry, Rz = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    Rx[1, 2], Rx[2, 1] = -1, 1
    Ry[0, 2], Ry[2, 0] = 1, -1
    Rz[0, 1], Rz[1, 0] = -1, 1
    coords = domain.get_node_position()
    rbm_rx = (Rx @ (coords[:,nod_idx] - com[:, None])).T.flatten()
    rbm_ry = (Ry @ (coords[:,nod_idx] - com[:, None])).T.flatten()
    rbm_rz = (Rz @ (coords[:,nod_idx] - com[:, None])).T.flatten()
    dof_idx = domain.get_dofnumber(nod_idx, dof_idx=[0, 1, 2]).flatten()
    assert np.allclose(rbm[dof_idx, 3], rbm_rx)
    assert np.allclose(rbm[dof_idx, 4], rbm_ry)
    assert np.allclose(rbm[dof_idx, 5], rbm_rz)
    # domain.write_to_vti(dict(rbm=rbm))


@pytest.mark.parametrize('n', [1, 2])
def test_element_offset(n: int):
    domain = pym.VoxelDomain(10, 10, unitx=0.1, unity=0.1)
    e = domain.elements[0, 5]
    e1 = domain.offset_element_set(e, n)
    e2 = domain.offset_element_set(e1, -n)
    plot = False
    if plot:
        fig, ax = plt.subplots(1, 3)
        x0 = np.zeros(domain.nel, dtype=bool)
        x0[e] = True

        coords = domain.get_node_position()[:, domain.nodes[..., 0]]
        ax[0].pcolor(coords[0], coords[1], x0[domain.elements[..., 0]].T, edgecolors='k', linewidths=1)
        ax[0].set_aspect('equal', adjustable='box')
        x1 = np.zeros(domain.nel, dtype=bool)
        x1[e1] = True
        ax[1].pcolor(coords[0], coords[1], x1[domain.elements[..., 0]].T, edgecolors='k', linewidths=1)
        ax[1].set_aspect('equal', adjustable='box')
        x2 = np.zeros(domain.nel, dtype=bool)
        x2[e2] = True
        ax[2].pcolor(coords[0], coords[1], x2[domain.elements[..., 0]].T, edgecolors='k', linewidths=1)
        ax[2].set_aspect('equal', adjustable='box')
        plt.show(block=True)

    assert all(np.sort(e1) == np.sort(domain.elements[0:n+1, 5-n:6+n].flatten()))
    assert all(np.sort(e2) == np.sort(e.flatten()))


if __name__ == '__main__':
    pytest.main([__file__])
    