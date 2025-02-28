""" Minimal example for a compliance topology optimization with multigrid preconditioned CG as solver """
import numpy as np

import pymoto as pym

nx, ny, nz = 32, 64, 64  # Set nz to zero for the 2D problem, nz > 0 runs a 3D problem
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

    # Make force and design vector, and fill with initial values
    sf = pym.Signal('f', state=f)
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    func = pym.Network(print_timing=True)

    # Filter
    sxfilt = func.append(pym.FilterConv(sx, domain=domain, radius=filter_radius))
    sx_analysis = sxfilt

    # Show the design on the screen as it optimizes (plotting domain in 3D may take significant time)
    # func.append(pym.PlotDomain(sx_analysis, domain=domain, saveto="out/design", clim=[0, 1]))

    # SIMP material interpolation
    sSIMP = func.append(pym.MathGeneral(sx_analysis, expression=f"{xmin} + {1.0 - xmin}*inp0^3"))

    # System matrix assembly module
    if thermal:
        sK = func.append(pym.AssemblePoisson(sSIMP, domain=domain, bc=boundary_dofs))
    else:
        sK = func.append(pym.AssembleStiffness(sSIMP, domain=domain, bc=boundary_dofs))

    # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
    solver = None  # Default (solver is automatically chosen based on matrix properties)
    # solver = pym.solvers.SolverSparsePardiso()  # Requires Intel MKL installed
    # solver = pym.solvers.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
    # solver = pym.solvers.SolverSparseCholeskyScikit()  # Requires scikit installed

    ''' Iterative solver: CG with Geometric Multi-grid preconditioning '''
    # Set up fine level
    mg1 = pym.solvers.GeometricMultigrid(domain)
    mgs = [mg1]
    n_levels = 5  # Number of levels (including fine grid)
    for i in range(n_levels - 1):  # Setup coarse levels
        mgs.append(pym.solvers.GeometricMultigrid(mgs[i].sub_domain))
        mgs[i].inner_level = mgs[i+1]

    # Set up the solver (comment out to use the default factorization, try this to see the difference in time)
    solver = pym.solvers.CG(preconditioner=mg1, verbosity=1, tol=1e-5)

    ''' From here the rest of the example is identical to ex_compliance.py '''
    su = func.append(pym.LinSolve([sK, sf], hermitian=True, solver=solver))

    # Output the design, deformation, and force field to a Paraview file
    func.append(pym.WriteToVTI([sx_analysis, su, sf], domain=domain, saveto='out/dat.vti'))

    # Compliance calculation c = f^T u
    scompl = func.append(pym.EinSum([su, sf], expression='i,i->'))
    scompl.tag = 'compliance'

    # MMA needs correct scaling of the objective
    sg0 = func.append(pym.Scaling(scompl, scaling=100.0))
    sg0.tag = "objective"

    # Calculate the volume of the domain by adding all design densities together
    svol = func.append(pym.EinSum(sx_analysis, expression='i->'))
    svol.tag = 'volume'

    # Volume constraint
    sg1 = func.append(pym.MathGeneral(svol, expression=f'10*(inp0/{domain.nel} - {volfrac})'))
    sg1.tag = "volume constraint"

    # Maybe you want to check the design-sensitivities?
    do_finite_difference = False
    if do_finite_difference:
        pym.finite_difference(func, sx, [sg0, sg1], dx=1e-4)
        exit()

    func.append(pym.PlotIter([sg0, sg1]))  # Plot iteration history

    # Do the optimization with MMA
    # pym.minimize_mma(func, [sx], [sg0, sg1])

    # Do the optimization with OC
    pym.minimize_oc(func, sx, sg0)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum {'temperature' if thermal else 'displacement'} is {max(np.absolute(su.state))}")
