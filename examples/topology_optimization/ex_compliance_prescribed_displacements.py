"""
Example of the design for stiffness of a structure subjected to prescribed displacements
 using topology optimization with:
(i) maximum stiffness between prescribed displacements and support, and
(ii) constrained by maximum volume.

Reference:

Koppen, S., Langelaar, M., & van Keulen, F. (2022). 
A simple and versatile topology optimization formulation for flexure synthesis. 
Mechanism and Machine Theory, 172, 104743.
DOI: http://dx.doi.org/10.1016/j.mechmachtheory.2022.104743
"""

import numpy as np

# flake8: noqa
import pymoto as pym

# Problem settings
nx, ny = 40, 40  # Domain size
xmin, filter_radius, volfrac = 1e-9, 2, 0.5  # Density settings
nu, E = 0.3, 1.0  # Material properties

if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    nodes_right = domain.get_nodenumber(nx, np.arange(ny + 1))
    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_left_horizontal = dofs_left[0::2]
    dofs_left_vertical = dofs_left[1::2]

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_left_horizontal, dofs_right, dofs_left_vertical]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup solution vectors and rhs
    ff = np.zeros_like(free_dofs, dtype=float)
    u = np.zeros_like(all_dofs, dtype=float)

    u[dofs_left_vertical] = 1.0
    up = u[prescribed_dofs]

    # Initial design
    signal_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    network = pym.Network()

    # Density filtering
    signal_filtered_variables = network.append(pym.DensityFilter(signal_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    signal_penalized_variables = network.append(pym.MathGeneral(signal_filtered_variables, expression=f"{xmin} + {1-xmin}*inp0^3"))

    # Assembly
    signal_stiffness = network.append(pym.AssembleStiffness(signal_penalized_variables, domain=domain, e_modulus=E, poisson_ratio=nu))

    # Solve system of equations
    up = pym.Signal('up', state=up)
    ff = pym.Signal('ff', state=ff)
    signal_state = network.append(pym.SystemOfEquations([signal_stiffness, ff, up], free=free_dofs, prescribed=prescribed_dofs))

    # Calculate compliance value
    signal_compliance = network.append(pym.EinSum([signal_state[0], signal_state[1]], expression='i,i->'))

    # Objective function
    signal_objective = network.append(pym.Scaling([signal_compliance], scaling=-1))
    signal_objective.tag = "Objective"

    # Volume
    signal_volume = network.append(pym.EinSum(signal_filtered_variables, expression='i->'))
    signal_volume.tag = "volume"

    # Volume constraint
    signal_volume_constraint = network.append(pym.Scaling(signal_volume, scaling=10.0, maxval=volfrac*domain.nel))
    signal_volume_constraint.tag = "Volume constraint"

    # Plotting
    network.append(pym.PlotDomain(signal_filtered_variables, domain=domain, saveto="out/design"))

    opt_responses = [signal_objective, signal_volume_constraint]  # Optimization responses
    network.append(pym.PlotIter(opt_responses))

    # Run optimization
    pym.minimize_mma(network, [signal_variables], opt_responses, verbosity=2)
