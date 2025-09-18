import pytest
import numpy as np
import pymoto as pym
import numpy.testing as npt


def fd_testfn(x0, dx, df_an, df_fd):
    npt.assert_allclose(df_an, df_fd, rtol=1e-7, atol=1e-5)


class MatrixSum(pym.Module):
    def __call__(self, A):
        return A.sum()

    def _sensitivity(self, dAsum):
        A = self.get_input_states()
        return pym.DyadCarrier(np.ones(A.shape[0]), np.ones(A.shape[1]), shape=A.shape) * dAsum


class TestAssembleStiffness:
    def test_rows_columns(self):
        """ Check if element rows and columns are implemented correctly """
        domain = pym.DomainDefinition(1, 1)
        elmat = np.arange(4*4).reshape((4, 4))

        s_x = pym.Signal('x', state=np.ones(domain.nel))

        # Assemble stiffness matrix
        sA = pym.AssembleGeneral(domain=domain, element_matrix=elmat)(s_x)

        npt.assert_allclose(sA.state.toarray(), elmat)

        sAsum = MatrixSum()(sA)
        pym.finite_difference(s_x, sAsum, test_fn=fd_testfn)

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
        s_K = pym.AssembleStiffness(domain=domain, bc=dofidx_left, e_modulus=E, poisson_ratio=nu, plane='stress')(s_x)

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

        sAsum = MatrixSum()(s_K)
        pym.finite_difference(s_x, sAsum, test_fn=fd_testfn)

    def test_FEA_pure_tensile_3d_one_element(self):
        Lx, Ly, Lz = 0.1, 0.2, 0.3
        domain = pym.DomainDefinition(1, 1, 1, unitx=Lx, unity=Ly, unitz=Lz)
        nodidx_left = domain.get_nodenumber(*np.meshgrid(0, range(domain.nely + 1), range(domain.nelz + 1))).flatten()
        # Fixed at (0,0,0), roller in z-direction at (0, 1, 0), roller in y-direction at (0, 0, 1)
        nod_00 = domain.get_nodenumber(0, 0, 0)
        nod_10 = domain.get_nodenumber(0, 1, 0)
        nod_01 = domain.get_nodenumber(0, 0, 1)
        dofidx_left = np.concatenate([nodidx_left * 3, 
                                      np.array([nod_00, nod_01]) * 3 + 1, 
                                      np.array([nod_00, nod_10]) * 3 + 2])
        nodidx_right = domain.get_nodenumber(*np.meshgrid(1, range(domain.nely + 1), range(domain.nelz + 1))).flatten()

        E, nu = 210e+9, 0.3

        s_x = pym.Signal('x', state=np.ones(domain.nel))

        # Assemble stiffness matrix
        s_K = pym.AssembleStiffness(domain=domain, bc=dofidx_left, e_modulus=E, poisson_ratio=nu)(s_x)

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

        sAsum = MatrixSum()(s_K)
        pym.finite_difference(s_x, sAsum, test_fn=fd_testfn)


class TestAssembleMass:
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
        s_M = pym.AssembleMass(domain=domain, material_property=rho, ndof=domain.dim)(s_x)

        npt.assert_allclose(s_M.state.toarray(), MEhc)

        sAsum = MatrixSum()(s_M)
        pym.finite_difference(s_x, sAsum, test_fn=fd_testfn)

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
        s_C = pym.AssembleMass(domain=domain, material_property=rho*cp, ndof=1)(s_x)

        npt.assert_allclose(s_C.state.toarray(), CEhc)

        sAsum = MatrixSum()(s_C)
        pym.finite_difference(s_x, sAsum, test_fn=fd_testfn)

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
        s_M = pym.AssembleMass(domain=domain, material_property=rho, ndof=domain.dim)(s_x)

        npt.assert_allclose(s_M.state.toarray(), MEhc)

        sAsum = MatrixSum()(s_M)
        pym.finite_difference(s_x, sAsum, test_fn=fd_testfn)


class TestAssemblePoisson:
    def test_conductivity_mat_2d(self):
        N = 1
        Lx, Ly, Lz = 2, 3, 4
        lx, ly, lz = Lx/N, Ly/N, Lz
        domain = pym.DomainDefinition(N, N, unitx=lx, unity=ly, unitz=lz)
        nodidx_left = domain.get_nodenumber(0, np.arange(domain.nely + 1))
        nodidx_right = domain.get_nodenumber(domain.nelx, np.arange(domain.nely + 1))
        kt = 1.0

        s_x = pym.Signal('x', state=np.ones(domain.nel))
        s_KT = pym.AssemblePoisson(domain=domain, bc=nodidx_left, material_property=kt)(s_x)

        q = np.zeros(domain.nnodes)
        Q = 1.0
        q[nodidx_right] = Q / nodidx_right.size

        # check with simple 1D heat conduction through wall Q = -k (dT/dx)
        T_chk = Q*Lx/(kt*Ly*Lz)
        T = np.linalg.solve(s_KT.state.toarray(), q)

        # Check if left boundary has T=0 and right boundary has T=T_chk
        npt.assert_allclose(T[nodidx_left], 0, atol=1e-10)
        npt.assert_allclose(T[nodidx_right], T_chk, rtol=1e-10)

        sAsum = MatrixSum()(s_KT)
        pym.finite_difference(s_x, sAsum, test_fn=fd_testfn)

    def test_conductivity_mat_3d(self):
        N = 1
        Lx, Ly, Lz = 2, 3, 4
        lx, ly, lz = Lx / N, Ly / N, Lz / N
        domain = pym.DomainDefinition(N, N, N, unitx=lx, unity=ly, unitz=lz)
        nodidx_left = domain.get_nodenumber(*np.meshgrid(0, range(domain.nely + 1), range(domain.nelz + 1))).flatten()
        nodidx_right = domain.nodes[-1, :, :].flatten()
        kt = 1.0

        s_x = pym.Signal('x', state=np.ones(domain.nel))
        s_KT = pym.AssemblePoisson(domain=domain, bc=nodidx_left, material_property=kt)(s_x)

        q = np.zeros(domain.nnodes)
        Q = 1.0
        q[nodidx_right] = Q / nodidx_right.size

        # check with simple 1D heat conduction through wall
        T_chk = Q * Lx / (kt * Ly * Lz)
        T = np.linalg.solve(s_KT.state.toarray(), q)

        # Check if left boundary has T=0 and right boundary has T=T_chk
        npt.assert_allclose(T[nodidx_left], 0, atol=1e-10)
        npt.assert_allclose(T[nodidx_right], T_chk, rtol=1e-10)

        sAsum = MatrixSum()(s_KT)
        pym.finite_difference(s_x, sAsum, test_fn=fd_testfn)


if __name__ == '__main__':
    pytest.main([__file__])