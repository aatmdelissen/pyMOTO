"""
Compliance minimization
=======================

This example demonstrates how to minimize the compliance of a 2D or 3D structure, using pyMOTO. Two different physics
are implemented: a static mechanical analysis (= stiffness maximization) and a static thermal analysis (= conductivity 
maximization). The example contains the essential components needed to run a topology optimization problem, including a 
density filter, material interpolation, finite-element assembly, solution to the linear system of equations, and an 
optimizer (OC or MMA).
"""
import numpy as np

import pymoto as pym

nx, ny, nz = 120, 40, 0  # Set nz to zero for the 2D problem, nz > 0 runs a 3D problem
xmin = 1e-9
filter_radius = 2.0
volfrac = 0.5
thermal = True  # True = Static thermal analysis; False = Static mechanical analysis will be done

if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D analysis
        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        if thermal:
            ndof = 1  # Number of dofs per node
            # Get dof numbers at the boundary
            boundary_dofs = domain.get_nodenumber(0, np.arange(ny // 4, (ny + 1) - ny // 4))

            # Make a force vector
            force_dofs = domain.get_nodenumber(*np.meshgrid(np.arange(1, nx + 1), np.arange(ny + 1)))

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

    # Make signal for design vector, and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    # with pym.Network(print_timing=False) as func:
    # Filter
    sxfilt = pym.DensityFilter(domain, radius=filter_radius)(sx)
    sx_analysis = sxfilt

    # Show the design on the screen as it optimizes
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
    su = pym.LinSolve(hermitian=True, solver=solver)(sK, f)

    # # Output the design, deformation, and force field to a Paraview file
    # pym.WriteToVTI(domain, saveto='out/dat.vti')(sx_analysis, su, f)

    # Compliance calculation c = f^T u
    scompl = pym.EinSum('i, i->')(su, f)
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
        pym.finite_difference(sx, [sg0, sg1], dx=1e-4)
        exit()

    pym.PlotIter()(sg0, sg1)  # Plot iteration history

    # Do the optimization with MMA
    # pym.minimize_mma(func, [sx], [sg0, sg1])

    # Do the optimization with OC
    pym.minimize_oc(sx, sg0)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum {'temperature' if thermal else 'displacement'} is {max(np.absolute(su.state))}")
