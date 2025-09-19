import pytest
import numpy as np
import pymoto as pym
import numpy.testing as npt

np.random.seed(0)


class TestStaticCondensation:
    def test_real_symmetric(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        N = 20
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = dofs_right
        main_dofs = dofs_left[0::2]
        free_dofs = np.setdiff1d(all_dofs, np.unique(np.hstack([main_dofs, prescribed_dofs])))

        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = pym.AssembleStiffness(domain)(sx)
        su = pym.StaticCondensation(free=free_dofs, main=main_dofs)(sK)
        sc = pym.EinSum('ij->')(su)
        sc.tag = 'c'
        
        # Check result
        sK1 = pym.AssembleStiffness(domain=domain, bc=np.concatenate([main_dofs[1:], prescribed_dofs]))(sx)
        sf = pym.Signal('f', np.zeros(domain.nnodes*2))
        sf.state[main_dofs[0]] = 1.0
        su1 = pym.LinSolve()(sK1, sf)
        suKu1 = pym.EinSum('i,i->')(su1, sf)

        npt.assert_allclose(suKu1.state, 1/su.state[0, 0])

        def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)

        pym.finite_difference(sx, sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)


if __name__ == '__main__':
    pytest.main([__file__])
