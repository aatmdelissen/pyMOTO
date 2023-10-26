"""
Example of the design of a compliant mechanism using topology optimization with:
(i) maximum stiffness of input and output ports, and
(ii) desired geometric advantage of -1
(iii) desired maximum stiffness of the compliant deformation pattern (inversion)

References:
Koppen, S. (2022).
Topology optimization of compliant mechanisms with multiple degrees of freedom.
DOI: http://dx.doi.org/10.4233/uuid:21994a92-e365-4679-b6ac-11a2b70572b7
"""
import numpy as np

# flake8: noqa
import pymoto as pym

# Problem settings
nx, ny = 100, 100  # Domain size
xmin, filter_radius, volfrac = 1e-6, 2, 0.35  # Density settings
nu, E = 0.3, 100  # Material properties

scaling_objective = 100.0

compliance_constraint_value = 0.1
scaling_compliance_constraint = 10.0

use_volume_constraint = True
scaling_volume_constraint = 10.0

v = np.ones(2, dtype=float)
v[1] *= -1.1  # intended geometric advantage

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
    dof_input = dofs_left_y[0]  # Input dofs for mechanism
    dof_output = dofs_left_y[-1]  # Output dofs for mechanism

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_left_x, dofs_right, dof_input, dof_output]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup solution vectors and rhs for two loadcases
    ff = np.zeros((free_dofs.size, 2), dtype=float)
    u = np.zeros((2 * domain.nnodes, 2), dtype=float)
    u[dof_input, :] = 1.0
    u[dof_output, 0] = 1.0
    u[dof_output, 1] = -1.0

    up = u[prescribed_dofs, :]
    prescribed_dofs = np.unique(np.hstack([dofs_left_x, dofs_right]))
    gdi = np.unique(np.hstack([dof_input, dof_output]))
    free_dofs = np.setdiff1d(all_dofs, np.unique(np.hstack([gdi, prescribed_dofs])))

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

    # Static condensation
    signal_condensed_stiffness = network.append(pym.StaticCondensation(signal_stiffness, main=gdi, free=free_dofs))

    # Compliances
    s_v = pym.Signal('u', state=v)
    signal_state = network.append(pym.EinSum([s_v, signal_condensed_stiffness, s_v], expression='i,ik,k->'))

    # compliance constraint
    signal_compliance_constraint = network.append(
        pym.Scaling(signal_state, scaling=scaling_compliance_constraint, maxval=compliance_constraint_value))
    signal_compliance_constraint.tag = "Compliance constraint"

    # Compliances
    f = np.eye(2, dtype=float)
    s_f = pym.Signal('f', state=f)
    s_u = network.append(pym.LinSolve([signal_condensed_stiffness, s_f]))

    s_compliances = network.append(pym.EinSum([s_u, s_f], expression='ij,ij->'))

    # Objective function
    signal_objective = network.append(pym.Scaling([s_compliances], scaling=1.0 * scaling_objective))
    signal_objective.tag = "Objective"

    # Volume
    signal_volume = network.append(pym.EinSum(signal_filtered_variables, expression='i->'))
    signal_volume.tag = "Volume"

    # Volume constraint
    signal_volume_constraint = network.append(
        pym.Scaling(signal_volume, scaling=scaling_volume_constraint, maxval=volfrac * domain.nel))
    signal_volume_constraint.tag = "Volume constraint"

    # Plotting
    network.append(pym.PlotDomain(signal_filtered_variables, domain=domain, saveto="out/design"))

    opt_responses = [signal_objective, signal_compliance_constraint]
    if use_volume_constraint:
        opt_responses.append(signal_volume_constraint)
    network.append(pym.PlotIter(opt_responses))

    # Optimization
    pym.minimize_mma(network, [signal_variables], opt_responses, verbosity=2)
