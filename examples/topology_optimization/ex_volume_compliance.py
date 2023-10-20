"""
Example of the design of cantilever for minimum volume subjected to displacement constraint.

References:
    None? s.koppen@tudelft.nl
"""

import numpy as np

import pymoto as pym

# Problem settings
nx, ny = 80, 40  # Domain size
xmin, filter_radius, volfrac = 1e-6, 2, 1.0  # Density settings

scaling_objective = 10.0
displacement_constraint_value = 100
scaling_displacement_constraint = 10.0

if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(nx, ny // 2) + 1] = 1.0

    # Initial design
    s_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    fn = pym.Network()

    # Filtering
    s_filtered_variables = fn.append(pym.DensityFilter(s_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    s_penalized_variables = fn.append(pym.MathGeneral(s_filtered_variables, expression=f"{xmin} + {1 - xmin}*inp0^3"))

    # Assemble stiffness matrix
    s_K = fn.append(pym.AssembleStiffness(s_penalized_variables, domain=domain, bc=dofs_left))

    # Solve
    s_force = pym.Signal('f', state=f)
    s_displacement = fn.append(pym.LinSolve([s_K, s_force], pym.Signal('u')))

    # Volume
    s_volume = fn.append(pym.EinSum(s_filtered_variables, expression='i->'))

    # Objective
    s_objective = fn.append(pym.Scaling([s_volume], scaling=scaling_objective))
    s_objective.tag = "Objective"

    # Output displacement (is a complex value)
    s_compliance = fn.append(pym.EinSum([s_displacement, s_force], expression='i,i->'))

    # Displacement constraint
    s_compliance_constraint = fn.append(
        pym.Scaling(s_compliance, scaling=scaling_displacement_constraint, maxval=displacement_constraint_value))
    s_compliance_constraint.tag = "Displacement constraint"

    # Plotting
    module_plotdomain = pym.PlotDomain(s_filtered_variables, domain=domain, saveto="out/design")
    responses = [s_objective, s_compliance_constraint]
    module_plotiter = pym.PlotIter(responses)
    fn.append(module_plotdomain, module_plotiter)

    # Optimization
    pym.minimize_mma(fn, [s_variables], responses, verbosity=2)
