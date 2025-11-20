"""Transient thermal temperature minimization
==============================

Minimal example for transient thermal topology optimization

This example contains some specific modules used in transient problems

- :py:class:`pymoto.TransientSolve` To solve the transient thermal problem
- :py:class:`pymoto.AssembleMass` Used with `ndof=1` for thermal capacity matrix assembly
- :py:class:`pymoto.SeriesToVTI` Used to export the design and transient simulation of a specific iteration to a Paraview VTI.series file

References (2D case):
  M.J.B. Theulings, R. Maas, L. Noël, F. van Keulen, M. Langelaar
  Reducing time and memory requirements in topology optimization of transient problems
  International Journal for Numerical Methods in Engineering 125(14), 2024.
  DOI: 10.1002/nme.7461
"""

import numpy as np
import pymoto as pym

nx, ny, nz = 80, 120, 0  # Set nz to zero for the 2D problem, nz > 0 runs a 3D problem
Lx, Ly, Lz = 2/30, 0.1, 0
xmin = 1e-6
filter_radius = 3.0
volfrac = 0.5
end_time = 10 # End time [s], as this value decreases the optimizer will disconnect from the boundary
dt = 0.05
end_step = int(end_time/dt)
theta = 0.5  # time-stepping algorithm, 0.0 for forward Euler, 0.5 for Crank-Nicolson, 1.0 for backward Euler
ndof = 1

if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D analysis
        # Generate a grid
        domain = pym.DomainDefinition(nx, ny, unitx=Lx/nx, unity=Ly/ny)

        # Get dof numbers at the boundary
        boundary_dofs = domain.get_nodenumber(np.arange(nx+1), 0)

        # Make a force vector of three areas with a heat load in the middle of the domain
        h_xs = np.arange(nx//2 - nx//15, nx//2 + nx//15 + 1)
        h1_dofs = domain.get_nodenumber(*np.meshgrid(h_xs, np.arange(ny//4 - ny//30, ny//4 + ny//30 + 1))).flatten()
        h2_dofs = domain.get_nodenumber(*np.meshgrid(h_xs, np.arange(ny//2 - ny//30, ny//2 + ny//30 + 1))).flatten()
        h3_dofs = domain.get_nodenumber(*np.meshgrid(h_xs, np.arange(3*ny//4 - ny//30, 3*ny//4 + ny//30 + 1))).flatten()
        q = np.zeros((domain.nnodes * ndof, int(end_time / dt + 1)))
        q[h1_dofs, 2*end_step//3:] = 6e3 * dt / h1_dofs.size
        q[h2_dofs, end_step//3:] = 7.5e2 * dt / h2_dofs.size
        q[h3_dofs, :] = 1e3 * dt / h3_dofs.size
        heat_dofs = np.concatenate((h1_dofs, h2_dofs, h3_dofs))

    else:
        domain = pym.DomainDefinition(nx, ny, nz)
        boundary_nodes = domain.get_nodenumber(*np.meshgrid(0, range(ny + 1), range(nz + 1))).flatten()

        boundary_dofs = boundary_nodes
        xrange, yrange, zrange = np.arange(nx//2 - nx//8, nx//2 + nx//8 + 1), np.arange(ny//2 - ny//8, ny//2 + ny//8 + 1), np.arange(nz//2 - nz//8, nz//2 + nz//8 + 1),
        heat_dofs = domain.get_nodenumber(*np.meshgrid(xrange, yrange, zrange)).flatten()

        q = np.zeros((domain.nnodes * ndof, int(end_time / dt + 1)))
        q[heat_dofs, :] = 1.0  # Uniform heat input of 1.0W at all selected dofs

    if domain.nnodes > 1e+6:
        print("Too many nodes :(")  # Safety, to prevent overloading the memory in your machine
        exit()

    if theta == 0.0:
        a = 1/1000 # thermal diffusivity = k / (rho*c)
        assert dt < np.average(domain.element_size)**2/(2*a), "time step too large for forward Euler, numerical solution unstable"



    T_0 = np.zeros(domain.nnodes)

    # Make heat and design vector, and fill with initial values
    sq = pym.Signal('q', state=q)
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network

    # Filter
    sxfilt = pym.DensityFilter(domain, radius=filter_radius)(sx)
    sxfilt.tag = "Filtered design"
    sx_analysis = sxfilt

    # Show the design on the screen as it optimizes
    if domain.dim == 2:
        pym.PlotDomain(domain, saveto="out/design", clim=[0, 1])(sx_analysis)

    # SIMP material interpolation
    sSIMP = pym.MathGeneral(f"{xmin} + {1.0 - xmin}*inp0^3")(sx_analysis)

    # System matrix assembly module
    sK = pym.AssemblePoisson(domain, bc=boundary_dofs)(sSIMP)
    sC = pym.AssembleMass(domain, bc=boundary_dofs, ndof=ndof, material_property=1000)(sSIMP)

    # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
    solver = None  # Default (solver is automatically chosen based on matrix properties)
    # solver = pym.solvers.SolverSparsePardiso()  # Requires Intel MKL installed
    # solver = pym.solvers.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
    # solver = pym.solvers.SolverSparseCholeskyScikit()  # Requires scikit installed
    sT = pym.TransientSolve(dt=dt, end=end_time, x0=T_0, solver=solver)(sq, sK, sC)
    sT.tag = "temperatures"

    # Output the design, temperature, and heat field to a Paraview file
    pym.WriteToVTI(domain, saveto='out/dat.vti')(sx_analysis, sT[:, -1], sq[:, -1])

    # Output the design, temperature over time, and heat field over time to Paraview series
    pym.SeriesToVTI(domain, saveto='out/dat.vti', delta_t=dt, interval=2)(sx_analysis, sT, sq)

    # Total temperature at heat input Tt = sum(T[heat dofs]) for all time steps
    stemps = pym.EinSum('ij->')(sT[heat_dofs, :])
    stemps.tag = 'average temperatures'

    # MMA needs correct scaling of the objective
    sg0 = pym.Scaling(scaling=100.0)(stemps)
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
    pym.minimize_mma(sx, [sg0, sg1])

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The average temperature value obtained is {stemps.state / (end_step * heat_dofs.size)}")
    print(f"The maximum temperature is {np.max(sT.state)}")