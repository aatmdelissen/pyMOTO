import pathlib  # For importing files
import sys
import pytest

import numpy as np
import numpy.testing as npt
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
from scipy.io import mmread  # For importing files

import pymoto as pym


def test_cg_nullspace():
    """Test nullspace removal of CG for singular matrix """

    domain = pym.VoxelDomain(32, 32)

    K = pym.AssembleStiffness(domain)(np.ones(domain.nel))

    nullsp = domain.get_rigid_body_modes()

    # Set up multigrid
    mg1 = pym.solvers.GeometricMultigrid(domain, smoother=pym.solvers.SOR(), smooth_steps=3)
    mgs = [mg1]
    while ((mgs[-1].sub_domain.nelx % 2) == 0
           and (mgs[-1].sub_domain.nely % 2 == 0)
           and (mgs[-1].sub_domain.nelz % 2 == 0)
           and (mgs[-1].sub_domain.nel > 100)):
        mgs.append(pym.solvers.GeometricMultigrid(mgs[-1].sub_domain, smoother=pym.solvers.SOR(), smooth_steps=3))
        mgs[-2].inner_level = mgs[-1]
    n_levels = len(mgs) + 1  # Number of levels (including fine grid)
    print(f"Setup GMG with {n_levels} levels, finest level is {mgs[-1].sub_domain.size}")
    # Set up the solver (comment out to use the default factorization, try this to see the difference in time)
    cg = pym.solvers.CG(preconditioner=mg1, verbosity=5, tol=1e-5, orth_subspace=nullsp)
    cg.update(K)

    f = np.zeros(domain.nnodes*domain.dim)
    f[domain.nodes[0, 0, 0]*domain.dim] = 1.0
    u = cg.solve(f)

    npt.assert_allclose(u @ nullsp, 0, atol=1e-8)
    fs = f - np.linalg.solve(nullsp.T @ nullsp, nullsp.T @ f) @ nullsp.T
    resi = np.linalg.norm(K@ u - fs) / np.linalg.norm(fs)
    assert resi < 1e-5


if __name__ == '__main__':
    pytest.main([__file__])