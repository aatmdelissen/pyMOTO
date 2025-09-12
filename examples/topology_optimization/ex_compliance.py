"""Compliance minimization
==========================

This example demonstrates how to minimize the compliance of a 2D or 3D structure using pymoto

Two different physics are implemented: a static mechanical analysis (= stiffness maximization) and a static thermal 
analysis (= conductivity maximization). The example contains the essential modules needed to run a topology 
optimization problem:

- :py:class:`pymoto.DensityFilter` Filtering of the design (to prevent checkerboarding)
- :py:class:`pymoto.PlotDomain` Utility to show the design as it optimizes
- :py:class:`pymoto.MathGeneral` Evaluate mathematical expression for material interpolation (SIMP)
- :py:class:`pymoto.AssemblePoisson` Assemble finite element matrix for the thermal problem
- :py:class:`pymoto.AssembleStiffness` Assemble the finite element matrix for the mechanical problem
- :py:class:`pymoto.LinSolve` Calculates the displacements or temperatures, by solving the linear system of equations 
  :math:`\mathbf{Ku}=\mathbf{f}`
- :py:class:`pymoto.EinSum` To perform a dot product to calculate the compliance :math:`c = \mathbf{u}\cdot\mathbf{f}`
- :py:class:`pymoto.Scaling` Utility to scale the objective function (and constraints if required) for the MMA optimizer

And next to that some important functions that operate on the :py:class:`pymoto.Network`:

- :py:func:`pymoto.finite_difference` Finite difference function to check the sensitivities of the optimization problem
- :py:func:`pymoto.minimize_oc` Minimization algorithm using optimality criteria (OC) method
- :py:func:`pymoto.minimize_mma` Minimization algorithm using the method of moving asymptotes (MMA)
"""
import numpy as np

import pymoto as pym

nx, ny, nz = 120, 40, 0  # Set nz to zero for the 2D problem, nz > 0 runs a 3D problem
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
            boundary_dofs = domain.nodes[0, int(ny/4):-int(ny/4)].flatten()

            # Make a force vector
            force_dofs = domain.nodes[1:,:].flatten()

        else:  # Mechanical
            ndof = 2
            # Calculate boundary dof indices
            boundary_dofs = domain.get_dofnumber(domain.nodes[0, :], np.arange(ndof), ndof=ndof).flatten() 

            # Which dofs to put a force on? The 1 is added for a force in y-direction (x-direction would be zero)
            force_dofs = ndof * domain.nodes[nx, int(ny/2)] + 1

    else:
        domain = pym.DomainDefinition(nx, ny, nz)
        
        if thermal:
            ndof = 1
            boundary_dofs = domain.nodes[0, int(3*ny/8):-int(3*ny/8), int(3*nz/8):-int(3*nz/8)].flatten()
            force_dofs = domain.nodes[int(nx/4):, :, :].flatten()
        else:  # Mechanical
            ndof = 3
            boundary_dofs = domain.get_dofnumber(domain.nodes[0, :, :], np.arange(ndof), ndof=ndof).flatten()  
            force_dofs = ndof * domain.nodes[nx, :, int(nz/2)] + 2  # Z-direction

    # Generate a force vector
    f = np.zeros(domain.nnodes * ndof)
    f[force_dofs] = 1.0  # Uniform force of 1.0 at all selected dofs

    # Make signal for design vector, and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    # Filter
    sxfilt = pym.FilterConv(domain, radius=filter_radius, ymin_bc=0, ymax_bc=0, zmin_bc=0, zmax_bc=0)(sx)
    sxfilt.tag = "Filtered design"
    sx_analysis = sxfilt

    # Show the design on the screen as it optimizes
    if domain.dim == 2:
        pym.PlotDomain(domain, saveto="out/design", clim=[0, 1])(sx_analysis)

    # SIMP material interpolation
    sSIMP = pym.MathGeneral(f"{xmin} + {1.0 - xmin}*inp0^3")(sx_analysis)

    # System matrix assembly module
    if thermal:
        sK = pym.AssemblePoisson(domain, bc=boundary_dofs)(sSIMP)
    else:
        sK = pym.AssembleStiffness(domain, bc=boundary_dofs)(sSIMP)

    # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
    # The matrix properties 'symmetric' and 'positive_definite' are passed to the LinSolve module, which are optional 
    # but can improve the solver choice. This may increase the speed of solving the linear system.
    solver = None  # Default (solver is automatically chosen based on matrix properties)
    # solver = pym.solvers.SolverSparsePardiso()  # Requires Intel MKL installed
    # solver = pym.solvers.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
    # solver = pym.solvers.SolverSparseCholeskyScikit()  # Requires scikit installed
    su = pym.LinSolve(symmetric=True, positive_definite=True, solver=solver)(sK, f)

    # # Output the design, deformation, and force field to a Paraview file
    pym.WriteToVTI(domain, saveto='out/dat.vti')(sx_analysis, su, f)

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
    # pym.minimize_mma(sx, [sg0, sg1])

    # Do the optimization with OC
    pym.minimize_oc(sx, sg0)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum {'temperature' if thermal else 'displacement'} is {max(np.absolute(su.state))}")
