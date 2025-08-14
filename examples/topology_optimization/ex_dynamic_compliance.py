"""Dynamic compliance
=====================

Example of the design of cantilever for minimum dynamic compliance and shows usage of complex values

This example contains some specific modules used in dynamic problems

- :py:class:`pymoto.AssembleMass` To assemble the mass matrix
- :py:class:`pymoto.AddMatrix` For addition of mass and stiffness matrices (with complex components) to form the dynamic 
  stiffness matrix
- :py:class:`pymoto.ComplexNorm` To calculate the norm of a complex value, corresponding to amplitude of vibration

References:
    Silva, O. M., Neves, M. M., & Lenzi, A. (2019).
    A critical analysis of using the dynamic compliance as objective function in topology optimization of one-material 
    structures considering steady-state forced vibration problems.
    Journal of Sound and Vibration, 444, 1-20.
    DOI: https://doi.org/10.1016/j.jsv.2018.12.030
"""
import numpy as np
import pymoto as pym

# Problem settings
lx, ly = 1, 0.5  # Domain size
ny = 50
nx = int(lx/ly)*ny
unitx, unity = lx / nx, ly / ny
unitz = 1.0  # out-of-plane thickness

xmin, filter_radius, volfrac = 1e-6, 2, 0.49  # Density settings

E, nu, rho = 210e9, 0.3, 7860  # Properties of steel in SI units [Pa], [-], [kg/m3]

# fundamental eigenfreq is 372.6 Hz, above this frequency the optimization will not result in a nice design
omega = 370 * (2 * np.pi)

# force
force_magnitude = -9000

# Rayleigh damping parameters
alpha, beta = 1e-3, 1e-8


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny, unitx=unitx, unity=unity, unitz=unitz)

    # Node and dof groups
    nodes_left = domain.nodes[0, :]
    dofs_left = domain.get_dofnumber(nodes_left, [0, 1], ndof=2)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(nx, ny // 2) + 1] = force_magnitude

    # Design vector signal
    s_x = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    with pym.Network() as fn:
        # Density filtering
        s_xfilt = pym.DensityFilter(domain, radius=filter_radius)(s_x)
        s_xfilt.tag = 'Filtered density'

        # Plot the design
        pym.PlotDomain(domain, saveto="out/design")(s_xfilt)

        # SIMP penalization
        s_xsimp = pym.MathGeneral(f"{xmin} + {1 - xmin}*inp0^3")(s_xfilt)

        # Assembly of stiffness matrix
        s_K = pym.AssembleStiffness(domain, e_modulus=E, poisson_ratio=nu, bc=dofs_left)(s_xsimp)

        # Assemble mass matrix
        s_M = pym.AssembleMass(domain, bc=dofs_left, material_property=rho, ndof=2)(s_xsimp)

        # Calculate the eigenfrequencies only once
        calculate_eigenfrequencies = True
        if calculate_eigenfrequencies:
            s_eig, _ = pym.EigenSolve(hermitian=True, nmodes=3)(s_K, s_M) 
 
            eigfreq = np.sqrt(s_eig.state)
            print(f"Eigenvalues are {eigfreq} rad/s or {eigfreq / (2 * np.pi)} Hz")

        # Build dynamic stiffness matrix Z = K + iω(αM + βK) - ω^2 M
        s_Z = pym.AddMatrix()(1 + 1j*omega*beta, s_K, -omega**2 + 1j*omega*alpha, s_M)

        # Solve linear system of equations
        s_u = pym.LinSolve()(s_Z, f)

        # Output displacement (is a complex value)
        s_cdyn = pym.EinSum('i,i->')(s_u, f)

        # Absolute value (amplitude of response)
        s_ampl = pym.ComplexNorm()(s_cdyn)

        # Objective
        s_objective = pym.Scaling(scaling=100.0)(s_ampl)
        s_objective.tag = "Objective"

        # Volume
        s_volume = pym.EinSum('i->')(s_xfilt)

        # Volume constraint
        s_volume_constraint = pym.Scaling(scaling=10.0, maxval=volfrac * domain.nel)(s_volume)
        s_volume_constraint.tag = "Volume constraint"

        # List of optimization responses: first the objective and all others the constraints
        responses = [s_objective, s_volume_constraint]
        
        # Show iteration history
        pym.PlotIter()(*responses)

    # Optimization
    pym.minimize_mma(s_x, responses, verbosity=2)
