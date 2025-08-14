""" Multigrid preconditioned CG solver
======================================

Significant speedups can be achieved using CG with a multigrid preconditioner to solve the linear system of equations

The speedup enables large-scale topology optimization problems, which is achieved by projecting the finite element 
problem onto a series of coarser grids, and only factorizing the finite element matrix at the coarsest grid. The 
solution is then interpolated back to the fine grid. This process is used as preconditioner for the CG iterations,
enabling fast convergence and optimization in 3D.

This example is based on  :ref:`sphx_glr_auto_examples_topology_optimization_ex_compliance.py` and only differs in the 
solver used. The user has full control over the number of multigrid levels and their internal settings (see 
:py:class:`pymoto.solvers.GeometricMultigrid` and :py:class:`pymoto.solvers.CG` for more details).

References:
  Amir, O., Aage, N., & Lazarov, B. S. (2014). 
  On multigrid-CG for efficient topology optimization. 
  Structural and Multidisciplinary Optimization, 49(5), 815-829.
  DOI: https://doi.org/10.1007/s00158-013-1015-5
"""
import numpy as np
import pymoto as pym

# nx, ny, nz = 32, 64, 64  # Domain sizes (# elements). Set nz to zero for the 2D problem, nz > 0 runs a 3D problem
nx, ny, nz = 512, 256, 0  # 2D
xmin = 1e-9
filter_radius = 2.0
volfrac = 0.5
thermal = False  # True = Static thermal analysis; False = Static mechanical analysis will be done

if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D analysis
        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        if thermal:
            ndof = 1  # Number of dofs per node
            # Get dof numbers at the boundary
            boundary_dofs = domain.nodes[0, ny//4:((ny + 1) - ny // 4)].flatten()

            # Make a force vector
            force_dofs = domain.nodes[1:, :].flatten()

        else:  # Mechanical
            ndof = 2
            # Calculate boundary dof indices
            boundary_nodes = domain.get_nodenumber(0, np.arange(ny + 1))
            boundary_dofs = np.repeat(boundary_nodes * ndof, ndof, axis=-1) + np.tile(np.arange(ndof), len(boundary_nodes))

            # Which dofs to put a force on? The 1 is added for a force in y-direction (x-direction would be zero)
            force_dofs = ndof * domain.get_nodenumber(nx, ny // 2) + 1

    else:
        domain = pym.DomainDefinition(nx, ny, nz)
        boundary_nodes = domain.get_nodenumber(*np.meshgrid(0, range(ny + 1), range(nz + 1))).flatten()

        if thermal:
            boundary_dofs = boundary_nodes
            force_dofs = domain.get_nodenumber(*np.meshgrid(np.arange(1, nx+1), np.arange(ny + 1), np.arange(nz + 1))).flatten()
            ndof = 1

        else:  # Mechanical
            ndof = 3
            boundary_dofs = np.repeat(boundary_nodes * ndof, ndof, axis=-1) + np.tile(np.arange(ndof), len(boundary_nodes))
            force_dofs = ndof * domain.get_nodenumber(nx, ny // 2, nz // 2) + 2  # Z-direction

    if domain.nnodes > 1e+6:
        print("Too many nodes :(")  # Safety, to prevent overloading the memory in your machine
        exit()

    # Generate a force vector
    f = np.zeros(domain.nnodes * ndof)
    f[force_dofs] = 1.0  # Uniform force of 1.0 at all selected dofs

    # Make design vector and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    with pym.Network(print_timing=True) as func:
        # Filter
        sxfilt = pym.FilterConv(domain, radius=filter_radius)(sx)
        sx_analysis = sxfilt
        sx_analysis.tag = 'design'

        # Show the design on the screen as it optimizes (plotting domain in 3D may take significant time)
        if nz == 0:
            pym.PlotDomain(domain, saveto="out/design", clim=[0, 1])(sx_analysis)

        # SIMP material interpolation
        sSIMP = pym.MathGeneral(f"{xmin} + {1.0 - xmin}*inp0^3")(sx_analysis)

        # System matrix assembly module
        if thermal:
            sK = pym.AssemblePoisson(domain, bc=boundary_dofs)(sSIMP)
        else:
            sK = pym.AssembleStiffness(domain, bc=boundary_dofs)(sSIMP)

        # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
        solver = None  # Default (solver is automatically chosen based on matrix properties)
        # solver = pym.solvers.SolverSparsePardiso()  # Requires Intel MKL installed
        # solver = pym.solvers.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
        # solver = pym.solvers.SolverSparseCholeskyScikit()  # Requires scikit installed

        ''' Iterative solver: CG with geometric multigrid preconditioning '''
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

        print(f"Multigrid levels: {len(mgs)}")
        print(f"Coarsest grid size: {mgs[-1].sub_domain.nelx} x {mgs[-1].sub_domain.nely} " + (f"x {mgs[-1].sub_domain.nelz}" if nz > 0 else ""))

        # Set up the solver (comment out to use the default factorization, try this to see the difference in time)
        solver = pym.solvers.CG(preconditioner=mg1, verbosity=1, tol=1e-5)

        ''' From here the rest of the example is identical to ex_compliance.py '''
        su = pym.LinSolve(hermitian=True, solver=solver)(sK, f)

        # Output the design, deformation, and force field to a Paraview file
        pym.WriteToVTI(domain, saveto='out/dat.vti')(sx_analysis, su, f)

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
        sg1 = pym.MathGeneral(f'10*(inp0/{domain.nel} - {volfrac})')(svol)
        sg1.tag = "volume constraint"

    # Maybe you want to check the design-sensitivities?
    do_finite_difference = False
    if do_finite_difference:
        pym.finite_difference(sx, [sg0, sg1], function=func, dx=1e-4)
        exit()

    with func:
        pym.PlotIter()(sg0, sg1)  # Plot iteration history

    # Do the optimization with MMA
    pym.minimize_mma([sx], [sg0, sg1], function=func)

    # Do the optimization with OC
    # pym.minimize_oc(sx, sg0, function=func)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum {'temperature' if thermal else 'displacement'} is {max(np.absolute(su.state))}")
