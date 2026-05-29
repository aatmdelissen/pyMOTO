""" Showcase for 3D thermal heat-sink
=====================================

This example showcases the topology optimization of thermal loadcase for a heat-sink

The structure is subjected to a distributed heat load and the optimization seeks to minimize the thermal resistance.
As the problem is 2-fold symmetric, only a quarter of the domain needs analysis and optimization. With post-processing
the design can be mirrored to restore the complete domain (e.g. using Paraview).

This example is based on  :ref:`sphx_glr_auto_examples_topology_optimization_ex_compliance_multigrid.py`.
"""
import numpy as np
import pymoto as pym

n = 128  # Resolution (# elements). Use powers of 2 for good GMG performance (e.g. 4, 8, 16, ...)
nx, ny, nz = n, n, 2*n  # Domain sizes (# elements)
Lx = 0.1  # Length in (m), other dimensions are deduced from `ny` and `nz`
xmin = 1e-9  # Minimum density value
filter_radius = 3.0  # Filter radius
volfrac = 0.1  # Target volume fraction

if __name__ == "__main__":
    print(__doc__)

    # Setup domain
    h = Lx/nx  # Element size
    domain = pym.VoxelDomain(nx, ny, nz, unitx=h, unity=h, unitz=h)  # Setup domain object

    # BC heat sink at bottom
    boundary_dofs = domain.nodes[:-int(nx/2), :-int(ny/2), 0].flatten()

    # Load vector on top section of domain
    eidx_load = domain.elements[:, :, int(nz/4):].flatten()
    nidx_load = domain.conn[eidx_load, :]
    f = np.zeros(domain.nnodes)
    np.add.at(f, nidx_load, 1.0)
    f *= 1 / (4*np.sum(f))  # Normalize to 1/4W unit heat input (1W for the total symmetric domain)

    # Make design vector and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Padding elements for density filter (set densities to 0)
    delem = int(np.ceil(filter_radius))
    delem0 = delem - 1
    eidx_xmax = domain.elements[-delem0:, :, :].flatten()
    eidx_ymax = domain.elements[:, -delem0:, :].flatten()
    eidx_zmax = domain.elements[:, :, -delem0:].flatten()
    eidx_0 = np.unique(np.concatenate([eidx_xmax, eidx_ymax, eidx_zmax]))

    # Start building the modular network
    with pym.Network() as func:
        # Set elements at the edge to zero to prevent the design 'sticking' to the boundary
        sx0 = pym.SetValue(indices=eidx_0, value=0.0)(sx)

        # Density filter
        sxfilt = pym.FilterConv(domain, radius=filter_radius, xmax_bc=0, ymax_bc=0, zmax_bc=0)(sx0)
        sx_analysis = sxfilt
        sx_analysis.tag = 'design'

        # SIMP material interpolation
        sSIMP = pym.MathExpression(f"{xmin} + {1.0 - xmin}*inp0^3")(sx_analysis)

        # System matrix assembly module
        sK = pym.AssemblePoisson(domain, bc=boundary_dofs, material_property=237.)(sSIMP)

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
