import unittest
import numpy as np
import pymoto as pym
import numpy.testing as npt
from scipy.sparse import csc_matrix


class DynamicMatrix(pym.Module):
    alpha = 0.5
    beta = 0.5

    def _response(self, K, M, omega):
        return K + 1j * omega * (self.alpha * M + self.beta * K) - omega ** 2 * M

    def _sensitivity(self, dZ):
        K, M, omega = [s.state for s in self.sig_in]
        dK = np.real(dZ) - (omega * self.beta) * np.imag(dZ)
        dM = (-omega ** 2) * np.real(dZ) - (omega * self.alpha) * np.imag(dZ)
        dZrM = np.real(dZ).contract(M)
        dZiK = np.imag(dZ).contract(K)
        dZiM = np.imag(dZ).contract(M)
        domega = -self.beta * dZiK - self.alpha * dZiM - 2 * omega * dZrM
        return dK, dM, domega


class TestLinSolveModuleSparse(unittest.TestCase):
    # # ------------- Symmetric -------------
    def test_symmetric_real_compliance2d(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        N = 10  # Number of elements
        dom = pym.DomainDefinition(N, N)
        np.random.seed(0)
        xmin = 1e-4
        sx = pym.Signal('x', xmin + (1-xmin)*np.random.rand(dom.nel))
        fixed_nodes = dom.get_nodenumber(0, np.arange(0, N+1))
        bc = np.concatenate((fixed_nodes*2, fixed_nodes*2+1))
        # Setup different rhs types
        iforce_x = dom.get_nodenumber(N, np.arange(0, N + 1)) * 2  # Force in x-direction
        iforce_y = dom.get_nodenumber(N, np.arange(0, N + 1)) * 2 + 1  # Force in y-direction

        force_vecs = dict()

        # Single force
        f = np.zeros(dom.nnodes*2)
        f[iforce_x] = 1.0
        force_vecs['single_real'] = f

        # Multiple rhs
        f = np.zeros((dom.nnodes * 2, 2))
        f[iforce_x, 0] = 1.0
        f[iforce_y, 1] = 1.0
        force_vecs['multiple_real'] = f

        for k, f in force_vecs.items():
            with self.subTest(f"RHS-{k}"):
                sf = pym.Signal('f', f)

                fn = pym.Network()
                sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), dom, bc=bc))
                su = fn.append(pym.LinSolve([sK, sf], pym.Signal('u')))

                fn.response()

                self.assertTrue(np.allclose(sK.state@su.state, sf.state))  # Check residual
                # Check finite difference
                # def tfn(x0, dx, df_an, df_fd): np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
                def tfn(x0, dx, df_an, df_fd): npt.assert_allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
                pym.finite_difference(fn, [sx, sf], su, test_fn=tfn, dx=1e-6, tol=1e-4, verbose=False)

    def test_symmetric_real_compliance3d(self):
        """ Test symmetric real sparse matrix (compliance in 3D)"""
        N = 3  # Number of elements
        dom = pym.DomainDefinition(N, N, N)
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(dom.nel))
        jfix, kfix = np.meshgrid(np.arange(0, N+1), np.arange(0, N+1), indexing='ij')
        fixed_nodes = dom.get_nodenumber(0, jfix, kfix).flatten()
        bc = np.concatenate((fixed_nodes*3, fixed_nodes*3+1, fixed_nodes*3+2))
        iforce = dom.get_nodenumber(N, np.arange(0, N+1), np.arange(0, N+1))*3 + 1
        sf = pym.Signal('f', np.zeros(dom.nnodes*3))
        sf.state[iforce] = 1.0

        fn = pym.Network()
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), dom, bc=bc))
        su = fn.append(pym.LinSolve([sK, sf], pym.Signal('u')))

        fn.response()

        self.assertTrue(np.allclose(sK.state@su.state, sf.state))  # Check residual
        # Check finite difference
        # def tfn(x0, dx, df_an, df_fd): np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=2e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sf], su, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_symmetric_complex_dyncompliance2d(self):
        """ Test symmetric complex sparse matrix (dynamic compliance in 2D)"""
        N = 5  # Number of elements
        dom = pym.DomainDefinition(N, N)
        np.random.seed(0)
        sx = pym.Signal('x', np.random.rand(dom.nel))
        fixed_nodes = dom.get_nodenumber(0, np.arange(0, N+1))
        bc = np.concatenate((fixed_nodes*2, fixed_nodes*2+1))
        iforce = dom.get_nodenumber(N, np.arange(0, N+1))*2 + 1
        sf = pym.Signal('f', np.zeros(dom.nnodes*2))
        sf.state[iforce] = 1.0

        sOmega = pym.Signal('omega', 0.1)
        fn = pym.Network()
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), dom, bc=bc))
        sM = fn.append(pym.AssembleMass(sx, pym.Signal('M'), dom, bc=bc, ndof=dom.dim))
        sZ = fn.append(DynamicMatrix([sK, sM, sOmega], pym.Signal('Z')))

        su = fn.append(pym.LinSolve([sZ, sf], pym.Signal('u')))

        fn.response()

        # spspla.eigsh(sK.state, M=sM.state, k=6, sigma=0.0)

        self.assertTrue(np.allclose(sZ.state@su.state, sf.state))  # Check residual
        # Check finite difference
        # def tfn(x0, dx, df_an, df_fd): np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5)
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sf, sOmega], su, test_fn=tfn, dx=1e-7, tol=1e-4, verbose=False)


class TestAssemblyAddValues(unittest.TestCase):
    def test_finite_difference(self):
        np.random.seed(0)
        N = 2
        # Set up the domain
        domain = pym.DomainDefinition(N, N)
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_x = dofs_left[0::2]
        dofs_left_y = dofs_left[1::2]
        dof_input = dofs_left_y[0]  # Input dofs for mechanism
        dof_output = dofs_left_y[-1]  # Output dofs for mechanism

        prescribed_dofs = np.union1d(dofs_left_x, dofs_right)

        # Setup rhs for two loadcases
        f = np.zeros(domain.nnodes * 2, dtype=float)
        f[dof_input] = 1.0

        # Initial design
        sx = pym.Signal('x', np.random.rand(domain.nel))
        signal_force = pym.Signal('f', state=f)
        # Setup optimization problem
        network = pym.Network()

        # Assembly
        istiff = np.array([dof_input, dof_output])
        sstiff = np.array([10.0, 10.0])

        K_const = csc_matrix((sstiff, (istiff, istiff)), shape=(domain.nnodes * 2, domain.nnodes * 2))
        signal_stiffness = network.append(pym.AssembleStiffness(sx, domain=domain, bc=prescribed_dofs, add_constant=K_const))
        su = network.append(pym.LinSolve([signal_stiffness, signal_force], pym.Signal('u')))
        sc = network.append(pym.EinSum(su, expression='i->'))
        network.response()

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))

        pym.finite_difference(network, [sx], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_added_stiffness_on_ground(self):
        np.random.seed(0)
        N = 2
        # Set up the domain
        domain = pym.DomainDefinition(N, N)
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_x = dofs_left[0::2]
        dofs_left_y = dofs_left[1::2]
        dof_input = dofs_left_y[0]  # Input dofs for mechanism
        dof_output = dofs_left_y[-1]  # Output dofs for mechanism

        prescribed_dofs = np.union1d(dofs_left_x, dofs_right)

        # Setup rhs for two loadcases
        f = np.zeros(domain.nnodes * 2, dtype=float)
        f[dof_input] = 1.0

        # Initial design
        sx = pym.Signal('x', np.random.rand(domain.nel))
        signal_force = pym.Signal('f', state=f)
        # Setup optimization problem
        network = pym.Network()

        # Assembly
        istiff = np.array([dof_input, dof_output])
        sstiff = np.array([10.0, 10.0])

        K_const = csc_matrix((sstiff, (istiff, istiff)), shape=(domain.nnodes * 2, domain.nnodes * 2))
        signal_stiffness = network.append(
            pym.AssembleStiffness(sx, domain=domain, bc=prescribed_dofs, add_constant=K_const))
        su = network.append(pym.LinSolve([signal_stiffness, signal_force], pym.Signal('u')))
        sc = network.append(pym.EinSum(su, expression='i->'))

        network.response()

        network2 = pym.Network()
        # Assembly
        istiff = np.array([dof_input, dof_output, 4, 5])
        sstiff = np.array([10.0, 10.0, 100.0, 100.0])

        K_const1 = csc_matrix((sstiff, (istiff, istiff)), shape=(domain.nnodes * 2, domain.nnodes * 2))
        signal_stiffness = network2.append(
            pym.AssembleStiffness(sx, domain=domain, bc=prescribed_dofs, add_constant=K_const1))
        su = network2.append(pym.LinSolve([signal_stiffness, signal_force], pym.Signal('u')))
        sc2 = network2.append(pym.EinSum(su, expression='i->'))

        network2.response()

        npt.assert_allclose(sc.state, sc2.state)


class TestSystemOfEquations(unittest.TestCase):
    def test_sparse_symmetric_real_compliance2d_single_load(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        np.random.seed(0)
        N=10
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_horizontal = dofs_left[0::2]
        dofs_left_vertical = dofs_left[1::2]

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Setup solution vectors and rhs
        ff = np.zeros_like(free_dofs, dtype=float)
        ff[:] = np.random.rand(len(free_dofs))
        u = np.zeros_like(all_dofs, dtype=float)

        u[dofs_left_vertical] = np.random.rand(len(dofs_left_vertical))
        up = u[prescribed_dofs]

        sff = pym.Signal('ff', ff)
        sup = pym.Signal('up', up)

        fn = pym.Network()
        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), domain))
        su = fn.append(pym.SystemOfEquations([sK, sff, sup], free=free_dofs, prescribed=prescribed_dofs))
        sc = fn.append(pym.EinSum([su[0], su[1]], expression='i,i->'))
        fn.response()
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sff, sup], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_sparse_symmetric_real_compliance2d_single_load_u(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        np.random.seed(0)
        N = 10
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_horizontal = dofs_left[0::2]
        dofs_left_vertical = dofs_left[1::2]

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Setup solution vectors and rhs
        ff = np.zeros_like(free_dofs, dtype=float)
        ff[:] = np.random.rand(len(free_dofs))
        u = np.zeros_like(all_dofs, dtype=float)

        u[dofs_left_vertical] = np.random.rand(len(dofs_left_vertical))
        up = u[prescribed_dofs]

        sff = pym.Signal('ff', ff)
        sup = pym.Signal('up', up)

        fn = pym.Network()
        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), domain))
        su = fn.append(pym.SystemOfEquations([sK, sff, sup], free=free_dofs, prescribed=prescribed_dofs))
        sc = fn.append(pym.EinSum([su[0]], expression='i->'))
        fn.response()

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))

        pym.finite_difference(fn, [sx, sff, sup], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_sparse_symmetric_real_compliance2d_single_load_f(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        np.random.seed(0)
        N = 10
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_horizontal = dofs_left[0::2]
        dofs_left_vertical = dofs_left[1::2]

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Setup solution vectors and rhs
        ff = np.zeros_like(free_dofs, dtype=float)
        ff[:] = np.random.rand(len(free_dofs))
        u = np.zeros_like(all_dofs, dtype=float)

        u[dofs_left_vertical] = np.random.rand(len(dofs_left_vertical))
        up = u[prescribed_dofs]

        sff = pym.Signal('ff', ff)
        sup = pym.Signal('up', up)

        fn = pym.Network()
        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), domain))
        su = fn.append(pym.SystemOfEquations([sK, sff, sup], free=free_dofs, prescribed=prescribed_dofs))
        sc = fn.append(pym.EinSum([su[1]], expression='i->'))
        fn.response()

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))

        pym.finite_difference(fn, [sx, sff, sup], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_sparse_symmetric_real_compliance2d_single_multi_load(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        np.random.seed(0)
        N=10
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_horizontal = dofs_left[0::2]
        dofs_left_vertical = dofs_left[1::2]

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Setup solution vectors and rhs
        ff = np.zeros((len(free_dofs), 2), dtype=float)
        ff[:, :] = np.random.rand(np.shape(ff)[0], np.shape(ff)[1])
        u = np.zeros((len(all_dofs), 2), dtype=float)

        u[dofs_left_vertical, 0] = np.random.rand(len(dofs_left_vertical))
        u[dofs_left_horizontal, 1] = np.random.rand(len(dofs_left_vertical))
        up = u[prescribed_dofs, :]

        sff = pym.Signal('ff', ff)
        sup = pym.Signal('up', up)

        fn = pym.Network()
        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), domain))
        su = fn.append(pym.SystemOfEquations([sK, sff, sup], free=free_dofs, prescribed=prescribed_dofs))
        sc1 = fn.append(pym.EinSum([su[0][:, 0], su[1][:, 0]], expression='i,i->'))
        sc2 = fn.append(pym.EinSum([su[0][:, 1], su[1][:, 1]], expression='i,i->'))
        sc = fn.append(pym.MathGeneral([sc1, sc2], expression='inp0 + inp1'))
        fn.response()
        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))
        pym.finite_difference(fn, [sx, sff, sup], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_sparse_symmetric_real_compliance2d_single_multi_load_u(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        np.random.seed(0)
        N = 10
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_horizontal = dofs_left[0::2]
        dofs_left_vertical = dofs_left[1::2]

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Setup solution vectors and rhs
        ff = np.zeros((len(free_dofs), 2), dtype=float)
        ff[:, :] = np.random.rand(np.shape(ff)[0], np.shape(ff)[1])
        u = np.zeros((len(all_dofs), 2), dtype=float)

        u[dofs_left_vertical, 0] = np.random.rand(len(dofs_left_vertical))
        u[dofs_left_horizontal, 1] = np.random.rand(len(dofs_left_vertical))
        up = u[prescribed_dofs, :]

        sff = pym.Signal('ff', ff)
        sup = pym.Signal('up', up)

        fn = pym.Network()
        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), domain))
        su = fn.append(pym.SystemOfEquations([sK, sff, sup], free=free_dofs, prescribed=prescribed_dofs))
        sc = fn.append(pym.EinSum(su[0], expression='ij->'))
        fn.response()

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))

        pym.finite_difference(fn, [sx, sff, sup], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)

    def test_sparse_symmetric_real_compliance2d_single_multi_load_f(self):
        """ Test symmetric real sparse matrix (compliance in 2D)"""
        np.random.seed(0)
        N = 10
        # Set up the domain
        domain = pym.DomainDefinition(N, N)

        # node groups
        nodes_left = domain.get_nodenumber(0, np.arange(N + 1))
        nodes_right = domain.get_nodenumber(N, np.arange(N + 1))

        # dof groups
        dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), N + 1)
        dofs_left_horizontal = dofs_left[0::2]
        dofs_left_vertical = dofs_left[1::2]

        # free and prescribed dofs
        all_dofs = np.arange(0, 2 * domain.nnodes)
        prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Setup solution vectors and rhs
        ff = np.zeros((len(free_dofs), 2), dtype=float)
        ff[:, :] = np.random.rand(np.shape(ff)[0], np.shape(ff)[1])
        u = np.zeros((len(all_dofs), 2), dtype=float)

        u[dofs_left_vertical, 0] = np.random.rand(len(dofs_left_vertical))
        u[dofs_left_horizontal, 1] = np.random.rand(len(dofs_left_vertical))
        up = u[prescribed_dofs, :]

        sff = pym.Signal('ff', ff)
        sup = pym.Signal('up', up)

        fn = pym.Network()
        sx = pym.Signal('x', np.random.rand(domain.nel))
        sK = fn.append(pym.AssembleStiffness(sx, pym.Signal('K'), domain))
        su = fn.append(pym.SystemOfEquations([sK, sff, sup], free=free_dofs, prescribed=prescribed_dofs))
        sc = fn.append(pym.EinSum(su[1], expression='ij->'))
        fn.response()

        def tfn(x0, dx, df_an, df_fd): self.assertTrue(np.allclose(df_an, df_fd, rtol=1e-3, atol=1e-5))

        pym.finite_difference(fn, [sx, sff, sup], sc, test_fn=tfn, dx=1e-5, tol=1e-4, verbose=False)


if __name__ == '__main__':
    unittest.main()
