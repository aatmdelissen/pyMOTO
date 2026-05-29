""" Showcase for 3D MMB beam compliance
=======================================

This example showcases the topology optimization of a 3D MBB beam (*i.e.* bridge-like) structure

The structure is subjected to 3-point bending and the optimization seeks to maximize the stiffness.
As the problem is 2-fold symmetric, only a quarter of the domain needs analysis and optimization. With post-processing
the design can be mirrored to restore the complete domain (e.g. using Paraview).

This example is based on  :ref:`sphx_glr_auto_examples_topology_optimization_ex_compliance_multigrid.py`.
"""
import numpy as np
import pymoto as pym

n = 32  # Resolution (# elements). Use powers of 2 for good GMG performance (e.g. 4, 8, 16, ...)
nx, ny, nz = 4*n, n, 2*n  # Domain sizes (# elements)
Lx = 10.0  # Length in (m), other dimensions are deduced from `ny` and `nz`
xmin = 1e-9  # Minimum density value
filter_radius = 2.0  # Filter radius
volfrac = 0.1  # Target volume fraction

if __name__ == "__main__":
    print(__doc__)

    # Setup domain
    h = Lx/nx  # Element size
    domain = pym.VoxelDomain(nx, ny, nz, unitx=h, unity=h, unitz=h)  # Setup domain object
    ndof = 3  # Number of dofs per node

    # Symmetry BC in x-direction
    nidx_xmin = domain.nodes[0, :, :].flatten()
    bc_x = domain.get_dofnumber(nidx_xmin, dof_idx=0, ndof=ndof)

    # Symmetry BC in y-direction
    nidx_ymin = domain.nodes[:, 0, :].flatten()
    bc_y = domain.get_dofnumber(nidx_ymin, dof_idx=1, ndof=ndof)

    # Simple support in z-direction
    nidx_supp = domain.nodes[-1, :, 0].flatten()
    bc_z = domain.get_dofnumber(nidx_supp, dof_idx=2, ndof=ndof)

    # Combine all boundary conditions
    boundary_dofs = np.concatenate([bc_x, bc_y, bc_z])

    # Load vector on top in z-direction
    delem = int(np.ceil(filter_radius))
    eidx_force = domain.elements[:delem, :, -delem:].flatten()
    nidx_force = domain.conn[eidx_force, :].flatten()
    didx_force = domain.get_dofnumber(nidx_force, dof_idx=2, ndof=ndof)
    f = np.zeros(domain.nnodes * ndof)
    np.add.at(f, didx_force, 1.0)
    f *= -1 / (4*np.sum(f))  # Normalize to -1/4N unit force (-1N for the total symmetric domain)

    # Make design vector and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Padding elements for density filter (set densities to 0)
    delem0 = delem - 1
    eidx_xmax = domain.elements[-delem0:, :, :].flatten()
    eidx_ymax = domain.elements[:, -delem0:, :].flatten()
    eidx_zmin = domain.elements[:, :, :delem0].flatten()
    eidx_zmax = domain.elements[:, :, -delem0:].flatten()
    eidx_0 = np.unique(np.concatenate([eidx_xmax, eidx_ymax, eidx_zmin, eidx_zmax]))

    # Solid elements at force and simple-support location (set densities to 1)
    eidx_bc = domain.elements[-delem:, :, :delem].flatten()
    eidx_1 = np.unique(np.concatenate([eidx_force, eidx_bc]))

    # Start building the modular network
    with pym.Network() as func:
        # Set elements at the edge to zero to prevent the design 'sticking' to the boundary
        sx0 = pym.SetValue(indices=eidx_0, value=0.0)(sx)

        # Density filter
        sxfilt = pym.FilterConv(domain, radius=filter_radius, xmax_bc=0, ymax_bc=0, zmin_bc=0, zmax_bc=0)(sx0)

        # Set elements to one at location where the force acts and where the simple support is
        sx1 = pym.SetValue(indices=eidx_1, value=1.0)(sxfilt)
        sx_analysis = sx1
        sx_analysis.tag = 'design'

        # SIMP material interpolation
        sSIMP = pym.MathExpression(f"{xmin} + {1.0 - xmin}*inp0^3")(sx_analysis)

        # System matrix assembly module
        sK = pym.AssembleStiffness(domain, bc=boundary_dofs, e_modulus=70e+9, poisson_ratio=0.3)(sSIMP)

        # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
        solver = None  # Default (solver is automatically chosen based on matrix properties)
        # solver = pym.solvers.SolverSparsePardiso()  # Requires Intel MKL installed
        # solver = pym.solvers.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
        # solver = pym.solvers.SolverSparseCholeskyScikit()  # Requires scikit installed

        # Iterative solver: CG with geometric multigrid preconditioning
        # Set up fine level multigrid operator
        mg1 = pym.solvers.GeometricMultigrid(domain)
        mgs = [mg1] # List of multigrid levels, starting with the finest grid
        while True:  # Setup multiple sub-levels
            # Stop if any of the new dimensions is odd
            if any([nn % 2 != 0 for nn in mgs[-1].sub_domain.size]):
                break  

            # Stop if any of the coarse dimensions is smaller than 8
            if any(mgs[-1].sub_domain.size < 8):
                break  

            # Create a new coarser grid
            mgs.append(pym.solvers.GeometricMultigrid(mgs[-1].sub_domain))
            mgs[-2].inner_level = mgs[-1]  # Set the inner level to the next coarser grid

        print(f"Fine grid size: {domain.nelx} x {domain.nely} x {domain.nelz}")
        print(f"Multigrid levels: {len(mgs)}")
        print(f"Coarsest grid size: {mgs[-1].sub_domain.nelx} x {mgs[-1].sub_domain.nely} x {mgs[-1].sub_domain.nelz}")

        # Set up the solver (comment out to use the default factorization, try this to see the difference in time)
        solver = pym.solvers.CG(preconditioner=mg1, verbosity=0, tol=1e-5)

        su = pym.LinSolve(hermitian=True, solver=solver)(sK, f)

        # Output the design, deformation, and force field to a Paraview file
        pym.WriteToVTI(domain, saveto='out/dat.vti')(sx_analysis, su)

        # Compliance calculation c = f^T u
        scompl = pym.EinSum('i,i->')(su, f)
        scompl.tag = 'compliance'

        # MMA needs correct scaling of the objective
        sg0 = pym.Scaling(scaling=100.0)(scompl)
        sg0.tag = "objective"

        # Calculate the volume of the domain by adding all design densities together
        svol = pym.EinSum('i->')(sx_analysis)
        svol.tag = 'volume'

        # Volume constraint
        sg1 = pym.Scaling(scaling=10.0, maxval=volfrac*domain.nel)(svol)
        sg1.tag = "volume constraint"

        pym.PlotIter()(sg0, sg1)  # Plot iteration history

    # Do the optimization with MMA
    # pym.minimize_mma([sx], [sg0, sg1], function=func)

    # Do the optimization with OC
    pym.minimize_oc(sx, sg0, function=func)

    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum displacement is {max(np.absolute(su.state))}")
