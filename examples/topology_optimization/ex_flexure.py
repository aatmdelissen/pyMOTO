"""
Example of the design of a flexure using topology optimization with:
(i) maximum shear stiffness
(ii) constrainted maximum stiffness in axial stiffness
"""
import numpy as np

# flake8: noqa
import pymoto as pym

# Problem settings
nx, ny = 40, 60  # Domain size
xmin, filter_radius, volfrac = 1e-9, 2, 0.5  # Density settings
nu, E = 0.3, 100.0  # Material properties
compliance_constraint_value = 1.0
scaling_volume_constraint = 10.0
scaling_compliance_constraint = 1.0
scaling_objective = -10.0

if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    nodes_right = domain.get_nodenumber(nx, np.arange(ny + 1))

    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_left_x = dofs_left[0::2]
    dofs_left_y = dofs_left[1::2]

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_left, dofs_right]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup solution vectors and rhs for two loadcases
    ff = np.zeros((free_dofs.size, 2), dtype=float)
    u = np.zeros((2 * domain.nnodes, 2), dtype=float)
    u[dofs_left_x, 1] = 1.0
    u[dofs_left_y, 0] = 1.0

    up = u[prescribed_dofs, :]

    # Initial design
    signal_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    network = pym.Network()

    # Density filtering
    signal_filtered_variables = network.append(pym.DensityFilter(signal_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    signal_penalized_variables = network.append(
        pym.MathGeneral(signal_filtered_variables, expression=f"{xmin} + {1 - xmin}*inp0^3"))

    # Assembly
    signal_stiffness = network.append(
        pym.AssembleStiffness(signal_penalized_variables, domain=domain, e_modulus=E, poisson_ratio=nu))

    # Solve system of equations for the two loadcases
    up = pym.Signal('up', state=up)
    ff = pym.Signal('ff', state=ff)
    signal_state = network.append(
        pym.SystemOfEquations([signal_stiffness, ff, up], free=free_dofs, prescribed=prescribed_dofs))

    # Output displacement
    signal_output_displacement = network.append(
        pym.EinSum([signal_state[0][:, 0], signal_state[1][:, 0]], expression='i,i->'))

    # Objective function
    signal_objective = network.append(pym.Scaling([signal_output_displacement], scaling=scaling_objective))
    signal_objective.tag = "Objective"

    # compliancess
    signal_compliance = network.append(pym.EinSum([signal_state[0][:, 1], signal_state[1][:, 1]], expression='i,i->'))

    # compliance constraint input and output
    signal_compliance_constraint = network.append(
        pym.Scaling(signal_compliance, scaling=scaling_compliance_constraint, maxval=compliance_constraint_value))
    signal_compliance_constraint.tag = "Compliance constraint"

    # Volume
    signal_volume = network.append(pym.EinSum(signal_filtered_variables, expression='i->'))
    signal_volume.tag = "Volume"

    # Volume constraint
    signal_volume_constraint = network.append(
        pym.Scaling(signal_volume, scaling=scaling_volume_constraint, maxval=volfrac * domain.nel))
    signal_volume_constraint.tag = "Volume constraint"

    # Plotting
    network.append(pym.PlotDomain(signal_filtered_variables, domain=domain, saveto="out/design"))

    opt_responses = [signal_objective, signal_compliance_constraint, signal_volume_constraint]
    network.append(pym.PlotIter(opt_responses))

    # Optimization
    pym.minimize_mma(network, [signal_variables], opt_responses, verbosity=2)
