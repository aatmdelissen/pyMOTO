"""Minimal example for a transient thermal compliance topology optimization"""
import numpy as np

import pymoto as pym

nx, ny, nz = 60, 60, 0  # Set nz to zero for the 2D problem, nz > 0 runs a 3D problem
xmin = 1e-6
filter_radius = 3.0
volfrac = 0.4
end_time = 200 # End time [s], as this value increases the optimizer starts to connect to the boundaries
dt = 2.0
end_step = int(end_time/dt)
theta = 0.5  # time-stepping algorithm, 0.0 for forward Euler, 0.5 for Crank-Nicolson, 1.0 for backward Euler


if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D analysis
        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        ndof = 1  # Number of dofs per node
        # Get dof numbers at the boundary
        nodes_west = domain.get_nodenumber(0, np.arange(ny+1))
        nodes_east = domain.get_nodenumber(nx, np.arange(ny+1))
        nodes_north = domain.get_nodenumber(np.arange(ny+1), ny)
        nodes_south = domain.get_nodenumber(np.arange(ny+1), 0)
        boundary_dofs = np.concatenate((nodes_west, nodes_east, nodes_north, nodes_south))

        # Make a force vector of a heat load in the middle of the domain
        xrange, yrange = np.arange(nx//2 - nx//8, nx//2 + nx//8 + 1), np.arange(ny//2 - ny//8, ny//2 + ny//8 + 1)
        heat_dofs = domain.get_nodenumber(*np.meshgrid(xrange, yrange))

    else:
        domain = pym.DomainDefinition(nx, ny, nz)
        boundary_nodes = domain.get_nodenumber(*np.meshgrid(0, range(ny + 1), range(nz + 1))).flatten()

        boundary_dofs = boundary_nodes
        xrange, yrange, zrange = np.arange(nx//2 - nx//8, nx//2 + nx//8 + 1), np.arange(ny//2 - ny//8, ny//2 + ny//8 + 1), np.arange(nz//2 - nz//8, nz//2 + nz//8 + 1),
        heat_dofs = domain.get_nodenumber(*np.meshgrid(xrange, yrange, zrange)).flatten()
        ndof = 1

    if domain.nnodes > 1e+6:
        print("Too many nodes :(")  # Safety, to prevent overloading the memory in your machine
        exit()

    if theta == 0.0:
        a = 1.0 # thermal diffusivity = k / (rho*c)
        assert dt < np.average(domain.element_size)**2/(2*a), "time step too large for forward Euler, numerical solution unstable"

    q = np.zeros((domain.nnodes * ndof, int(end_time/dt + 1)))
    q[heat_dofs, :] = 1.0  # Uniform heat input of 1.0W at all selected dofs

    T_0 = np.zeros(domain.nnodes)

    # Make heat and design vector, and fill with initial values
    sq = pym.Signal('q', state=q)
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    func = pym.Network(print_timing=False)

    # Filter
    sxfilt = func.append(pym.DensityFilter(sx, domain=domain, radius=filter_radius))
    sx_analysis = sxfilt

    # Show the design on the screen as it optimizes
    func.append(pym.PlotDomain(sx_analysis, domain=domain, saveto="out/design", clim=[0, 1]))

    # SIMP material interpolation
    sSIMP = func.append(pym.MathGeneral(sx_analysis, expression=f"{xmin} + {1.0 - xmin}*inp0^3"))

    # System matrix assembly module
    sK = func.append(pym.AssemblePoisson(sSIMP, domain=domain, bc=boundary_dofs))
    sC = func.append(pym.AssembleMass(sSIMP, domain=domain, bc=boundary_dofs, ndof=ndof))

    # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
    solver = None  # Default (solver is automatically chosen based on matrix properties)
    # solver = pym.solvers.SolverSparsePardiso()  # Requires Intel MKL installed
    # solver = pym.solvers.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
    # solver = pym.solvers.SolverSparseCholeskyScikit()  # Requires scikit installed
    sT = func.append(pym.TransientSolve([sq, sK, sC], end=end_time, dt=dt, x0=T_0, solver=solver))
    sT.tag = "temperatures"

    # Output the design, temperature, and heat field to a Paraview file
    func.append(pym.WriteToVTI([sx_analysis, sT[:, -1], sq[:, -1]], domain=domain, saveto='out/dat.vti'))
    func.append(pym.TransientToVTI([sx_analysis, sT, sq], domain=domain, transient_tags=["temperatures", "q"], saveto='out/dat.vti', delta_t=2.0, interval=2))

    # Compliance calculation c = q^T T on final time step
    scompl = func.append(pym.EinSum([sq[:, -1], sT[:, -1]], expression='i,i->'))
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
    print(f"The maximum temperature is {np.max(sT.state)}")