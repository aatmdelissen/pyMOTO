"""
Example of the design of a compliant mechanism using topology optimization with:
(i) maximum output displacement due to input load
(ii) constrained minimum input and output stiffness
"""
# flake8: noqa
import pymoto as pym
import numpy as np

# Problem settings
nx, ny = 40, 40  # Domain size
xmin, filter_radius, volfrac = 1e-9, 2, 0.3  # Density settings
nu, E = 0.3, 1.0  # Material properties

input_compliance = 1000
output_compliance = 1000

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

    prescribed_dofs = np.union1d(dofs_left_x, dofs_right)

    # Setup rhs for two loadcases
    f = np.zeros((domain.nnodes*2, 2), dtype=float)
    f[dof_output, 0] = 1.0
    f[dof_input, 1] = 1.0

    # Initial design
    signal_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    network = pym.Network()

    # Filtering
    signal_filtered_variables = network.append(pym.DensityFilter(signal_variables, domain=domain, radius=filter_radius))

    # Penalization
    signal_penalized_variables = network.append(pym.MathGeneral(signal_filtered_variables, expression=f"{xmin} + {1-xmin}*inp0^3"))

    # Assembly
    signal_stiffness = network.append(pym.AssembleStiffness(signal_penalized_variables, domain=domain, e_modulus=E, poisson_ratio=nu, bc=prescribed_dofs))

    # Solve
    signal_force = pym.Signal('f', state=f)
    signal_displacements = network.append(pym.LinSolve([signal_stiffness, signal_force]))

    # Output displacement
    signal_output_displacement = network.append(pym.EinSum([signal_displacements[:, 1], signal_force[:, 0]], expression='i,i->'))

    # Objective
    signal_objective = network.append(pym.Scaling([signal_output_displacement], scaling=1))
    signal_objective.tag = "Objective"

    # Compliances
    signal_compliance = network.append(pym.EinSum([signal_displacements, signal_force], expression='ij,ij->j'))

    # Compliance constraint input and output
    signal_compliance_constraint_output = network.append(pym.Scaling(signal_compliance[0], scaling=10.0, maxval=output_compliance))
    signal_compliance_constraint_input = network.append(pym.Scaling(signal_compliance[1], scaling=10.0, maxval=input_compliance))
    signal_compliance_constraint_output.tag = "Output compliance constraint"
    signal_compliance_constraint_input.tag = "Input compliance constraint"

    # Volume
    signal_volume = network.append(pym.EinSum(signal_filtered_variables, expression='i->'))
    signal_volume.tag = "volume"

    # Volume constraint
    signal_volume_constraint = network.append(pym.Scaling(signal_volume, scaling=10.0, maxval=volfrac*domain.nel))
    signal_volume_constraint.tag = "Volume constraint"

    # Plotting
    module_plotdomain = pym.PlotDomain(signal_filtered_variables, domain=domain, saveto="out/design")
    responses = [signal_objective, signal_compliance_constraint_output, signal_compliance_constraint_input, signal_volume_constraint]
    module_plotiter = pym.PlotIter(responses)
    network.append(module_plotdomain, module_plotiter)

    # Optimization
    pym.minimize_mma(network, [signal_variables], responses, verbosity=2)
