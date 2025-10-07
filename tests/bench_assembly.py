# import pytest
import numpy as np
import pymoto as pym
import time


def setup_2d_domain(n):
    domain = pym.DomainDefinition(n, n)
    idx_bc = domain.get_dofnumber(domain.nodes[:, 0], ndof=2)
    idx_f = domain.get_dofnumber(domain.nodes[0, -1], 1, ndof=1)
    return domain, idx_bc, idx_f


def setup_3d_domain(n):
    domain = pym.DomainDefinition(n, n, n)
    idx_bc = domain.get_dofnumber(domain.nodes[:, 0, :], ndof=2)
    idx_f = domain.get_dofnumber(domain.nodes[0, -1, 0], 1, ndof=1)
    return domain, idx_bc, idx_f


def setup_matrix_solver(x, domain, idx_bc, idx_f, w=0.0, eta=0.0, xmin=1e-9, **kwargs):
    f = np.zeros(domain.nnodes*domain.dim)
    f[idx_f] = 1.0

    xfilt = pym.FilterConv(domain, radius=3)(x)
    xsimp = pym.MathGeneral(f'{xmin} + {1-xmin}*inp0^3')(xfilt)
    K = pym.AssembleStiffness(domain=domain, bc=idx_bc, **kwargs)(xsimp)
    mats = [1, K]

    if eta != 0:
        # Damping stiffness with other interpolation to ensure there is no linear dependency
        Kdamp = pym.AssembleStiffness(domain=domain, bc=idx_bc, bcdiagval=0, **kwargs)(xfilt)
        mats = [*mats, 1j*eta, Kdamp]

    if w != 0:
        M = pym.AssembleMass(domain, ndof=domain.dim, bc=idx_bc, **kwargs)(xfilt)
        mats = [*mats, -w**2, M]
    
    return pym.AddMatrix()(*mats)

def setup_matrix_multi(x, domain, idx_bc, idx_f, w=0.0, eta=0.0, xmin=1e-9, **kwargs):
    f = np.zeros(domain.nnodes*domain.dim)
    f[idx_f] = 1.0

    dummy_domain = pym.DomainDefinition(1, 1, 1 if domain.nelz > 0 else 0)

    xfilt = pym.FilterConv(domain, radius=3)(x)
    xsimp = pym.MathGeneral(f'{xmin} + {1-xmin}*inp0^3')(xfilt)

    elmat = [pym.AssembleStiffness(dummy_domain).elmat[0]]
    if eta != 0:
        # Damping stiffness with other interpolation to ensure there is no linear dependency
        elmat.append(1j*eta*pym.AssembleStiffness(dummy_domain).elmat[0])

    if w != 0:
        elmat.append(- w**2 * pym.AssembleMass(dummy_domain, ndof=domain.dim).elmat[0])

    return pym.AssembleGeneral(domain, elmat, bc=idx_bc, **kwargs)(xsimp, xfilt, xfilt)


def test_assembly(n=20, w=1, eta=0.1):
    np.random.seed(0)

    domain, idx_bc, idx_f = setup_3d_domain(n)

    # Generate random design
    sx = pym.Signal('x', np.random.rand(domain.nel))
   
    # Setup network
    start = time.perf_counter()
    with pym.Network() as fn:
        # _su = setup_matrix_solver(sx, domain, idx_bc, idx_f, w=w, eta=eta, reuse_sparsity=True)
        _su = setup_matrix_multi(sx, domain, idx_bc, idx_f, w=w, eta=eta, reuse_sparsity=True)
    elapsed = time.perf_counter() - start
    print(f"Network initialization, SETUP -- Elapsed time: {elapsed:0.4f} seconds")
    
    # Run many solves with different initial values, keep between 0 and 1
    move = 0.2
    n_test = 10

    start = time.perf_counter()
    for i in range(n_test):
        i_start = time.perf_counter()
        sx.state = np.clip(sx.state + move*(np.random.rand(domain.nel) - 0.5), 0, 1)
        fn.response()
        i_elapsed = time.perf_counter() - i_start
        print(f"Response {i} -- Elapsed time: {i_elapsed:0.4f} seconds")

    elapsed = time.perf_counter() - start
    print(f"Responses ({n_test}x) -- Elapsed time: {elapsed/n_test:0.4f} seconds average")


if __name__ == "__main__":
    test_assembly()
