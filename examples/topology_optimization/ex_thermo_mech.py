"""
Example of the design of a thermoelastic structure, with combined heat and mechanical load.

Implementation by @artofscience (s.koppen@tudelft.nl) based on:
Adapted by @JoranZwet to incorporate thermal analysis and expansion based on non-uniform temperature

Gao, T., & Zhang, W. (2010).
Topology optimization involving thermo-elastic stress loads.
Structural and multidisciplinary optimization, 42, 725-738.
DOI: https://doi.org/10.1007/s00158-010-0527-5
"""

import numpy as np
import pymoto as pym

# Problem settings
nx, ny = 60, 80  # Domain size
xmin, filter_radius = 1e-9, 2

load = -100.0  # point load
heatload = 1.0
scaling_objective = 10.0

volfrac = 0.25
scaling_volume_constraint = 1.0


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    nodes_right = domain.get_nodenumber(nx, np.arange(ny + 1))
    dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))

    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_left_x = dofs_left[0::2]

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_left_x, dofs_right]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(0, 0) + 1] = load
    q = np.zeros(domain.nnodes)  # Generate a heat vector
    q[domain.get_nodenumber(0, 0)] = heatload

    # Initial design
    s_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    fn = pym.Network()

    # Filtering
    s_filtered_variables = fn.append(pym.DensityFilter(s_variables, domain=domain, radius=filter_radius))
    s_filtered_variables.tag = "design"

    # RAMP with q = 2
    s_penalized_variables = fn.append(pym.MathGeneral(s_filtered_variables, expression=f"{xmin} + {1 - xmin}*(inp0 / (3 - 2*inp0))"))

    # Assemble stiffness and conductivity matrix
    s_K = fn.append(pym.AssembleStiffness(s_penalized_variables, domain=domain, bc=prescribed_dofs))
    s_KT = fn.append(pym.AssemblePoisson(s_penalized_variables, domain=domain, bc=nodes_right))

    # Solve for temperature
    s_heat = pym.Signal('q', state=q)
    s_temperature = fn.append(pym.LinSolve([s_KT, s_heat]))
    s_temperature.tag = "temperature"

    # Determine thermo-mechanical load
    s_TE = fn.append(pym.ElementAverage(s_temperature, domain=domain))
    s_xT = fn.append(pym.MathGeneral([s_TE, s_filtered_variables], expression="inp0 * inp1"))
    s_thermal_load = fn.append(pym.ThermoMechanical(s_xT, domain=domain, alpha=1.0))

    # Solve for displacements
    s_force = pym.Signal('f', state=f)
    s_load = fn.append(pym.MathGeneral([s_force, s_thermal_load], expression="inp0 + inp1"))

    s_up = pym.Signal('up', state=np.zeros(len(prescribed_dofs), dtype=float))
    s_disp, s_reactions = fn.append(pym.SystemOfEquations([s_K, s_load[free_dofs], s_up], prescribed=prescribed_dofs, free=free_dofs))
    s_disp.tag = "displacement"

    # Compliance
    s_compliance = fn.append(pym.EinSum([s_disp, s_reactions], expression='i,i->'))

    # Objective
    s_objective = fn.append(pym.Scaling([s_compliance], scaling=scaling_objective))
    s_objective.tag = "Objective"

    # Plotting
    fn.append(pym.PlotDomain(s_filtered_variables, domain=domain, saveto="out/design.png"))
    responses = [s_objective]

    # Output to VTK
    fn.append(pym.WriteToVTI([s_filtered_variables, s_temperature, s_disp], domain=domain, saveto="out/dat.vti"))

    # Volume
    s_volume = fn.append(pym.EinSum(s_filtered_variables, expression='i->'))

    # Volume constraint
    s_volume_constraint = fn.append(pym.Scaling(s_volume, scaling=scaling_volume_constraint, maxval=volfrac * domain.nel))
    s_volume_constraint.tag = "Volume constraint"
    responses.append(s_volume_constraint)

    fn.append(pym.PlotIter(responses))

    # Optimization
    pym.minimize_mma(fn, [s_variables], responses, verbosity=3, maxit=100)
