"""
Example of the design of a compliant mechanism using topology optimization with:
(i) maximum output displacement due to input load
(ii) constrained maximum volume
(iii) spring attached to output
(iv) (optional) spring attached to input

Note: this problem formulation only works with
(i) spring with positive output stiffness, and
(ii) non-design domains at the input and output, and
(iii) active volume constraint
(iv) input spring stiffness not equal to output spring stiffness

Implemented by @artofscience (s.koppen@tudelft.nl) based on:

Bendsoe, M. P., & Sigmund, O. (2003).
Topology optimization: theory, methods, and applications.
Springer Science & Business Media.
DOI: https://doi.org/10.1007/978-3-662-05086-6
"""
import numpy as np

# flake8: noqa
import pymoto as pym
from modules import VecSet

# Problem settings
nx, ny = 80, 80  # Domain size
xmin, filter_radius, volfrac = 1e-6, 2, 0.3  # Density settings
nu, E = 0.3, 100  # Material properties

scaling_objective = 10.0

input_spring_stiffness = 10
output_spring_stiffness = 10

use_volume_constraint = True
scaling_volume_constraint = 10.0

domain_size = 4


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
    f = np.zeros(domain.nnodes * 2, dtype=float)
    f[dof_input] = 1.0

    # Initial design
    signal_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    network = pym.Network()

    # Setup non-design domains
    input_domain = domain.get_elemnumber(*np.meshgrid(range(domain_size), range(domain_size)))
    output_domain = domain.get_elemnumber(*np.meshgrid(range(domain_size), np.arange(ny - domain_size, ny)))
    non_design_domain = np.union1d(input_domain, output_domain)
    signal_variables_2 = network.append(VecSet(signal_variables, indices=non_design_domain, value=1.0))

    # Filtering
    signal_filtered_variables = network.append(
        pym.DensityFilter(signal_variables_2, domain=domain, radius=filter_radius))

    # Penalization
    signal_penalized_variables = network.append(
        pym.MathGeneral(signal_filtered_variables, expression=f"{xmin} + {1 - xmin}*inp0^3"))

    # Assembly
    istiff = np.array([dof_input, dof_output])
    jstiff = np.copy(istiff)
    sstiff = np.array([input_spring_stiffness, output_spring_stiffness])

    signal_stiffness = network.append(
        pym.AssembleStiffness(signal_penalized_variables, domain=domain, e_modulus=E, poisson_ratio=nu,
                              bc=prescribed_dofs, add_stiffness=[istiff, jstiff, sstiff]))

    # Solve
    signal_force = pym.Signal('f', state=f)
    signal_displacements = network.append(pym.LinSolve([signal_stiffness, signal_force]))

    # Output displacement
    l = np.zeros_like(f)
    l[dof_output] = 1.0
    signal_selector = pym.Signal('l', state=l)
    signal_output_displacement = network.append(
        pym.EinSum([signal_displacements, signal_selector], expression='i,i->'))

    # Objective
    signal_objective = network.append(pym.Scaling([signal_output_displacement], scaling=scaling_objective))
    signal_objective.tag = "Objective"

    # Volume
    signal_volume = network.append(pym.EinSum(signal_filtered_variables, expression='i->'))
    signal_volume.tag = "volume"

    # Volume constraint
    signal_volume_constraint = network.append(pym.Scaling(signal_volume, scaling=10.0, maxval=volfrac * domain.nel))
    signal_volume_constraint.tag = "Volume constraint"

    # Plotting
    module_plotdomain = pym.PlotDomain(signal_filtered_variables, domain=domain, saveto="out/design")
    responses = [signal_objective]
    if use_volume_constraint:
        responses.append(signal_volume_constraint)
    module_plotiter = pym.PlotIter(responses)
    network.append(module_plotdomain, module_plotiter)

    # Optimization
    pym.minimize_mma(network, [signal_variables], responses, verbosity=2)
