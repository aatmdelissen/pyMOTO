"""Coupled thermo-mechanical loads
==================================

Example of the design of a thermoelastic structure with combined heat and mechanical load

First the heat equations are solved to determine the temperature distribution. After that, the mechanical load due to 
thermal expansion is calculated based on the temperatures. The compliance of the heat-expansion load combined with the 
mechanical load is minimized.

This example contains the following specific modules

- :py:class:`pymoto.AssemblePoisson` To assemble the conductivity matrix
- :py:class:`pymoto.AssembleStiffness` For assembly of the mechanical stiffness matrix
- :py:class:`pymoto.ElementAverage` Calculates the element average from nodal values (in this case temperature)
- :py:class:`pymoto.ThermoMechanical` Calculates mechanical loads based on thermal expansion
- :py:class:`pymoto.WriteToVTI` In this case used to export the design, temperatures, and deformations to Paraview

References:
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

load = -100.0  # Mechanical force
heatload = 1.0  # Thermal heat load

volfrac = 0.25


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

     # Node and dof groups
    nodes_right = domain.nodes[-1, :]
    nodes_left = domain.nodes[0, :]

    dofs_right = domain.get_dofnumber(nodes_right, [0, 1], ndof=2).flatten()
    dofs_left_x = domain.get_dofnumber(nodes_left, 0, ndof=2).flatten()
   
    fixed_dofs = np.unique(np.hstack([dofs_left_x, dofs_right]))

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.nodes[0, 0] + 1] = load
    q = np.zeros(domain.nnodes)  # Generate a heat vector
    q[domain.nodes[0, 0]] = heatload

    # Initial design
    s_x = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    with pym.Network() as fn:
        # Density filtering
        s_xfilt = pym.DensityFilter(domain, radius=filter_radius)(s_x)
        s_xfilt.tag = 'Filtered density'

        # Plot the design
        pym.PlotDomain(domain, saveto="out/design")(s_xfilt)

        # RAMP with q = 2
        s_xRAMP = pym.MathGeneral(f"{xmin} + {1 - xmin}*(inp0 / (3 - 2*inp0))")(s_xfilt)

        # Assemble stiffness and conductivity matrix
        s_K = pym.AssembleStiffness(domain, bc=fixed_dofs)(s_xRAMP)
        s_KT = pym.AssemblePoisson(domain, bc=nodes_right)(s_xRAMP)

        # Solve for temperature
        s_T = pym.LinSolve()(s_KT, q)
        s_T.tag = "temperature"

        # Determine thermo-mechanical load
        s_Telem = pym.ElementAverage(domain)(s_T)
        s_xT = pym.MathGeneral("inp0 * inp1")(s_Telem, s_xfilt)
        s_thermal_load = pym.ThermoMechanical(domain, alpha=1.0)(s_xT)

        # Combine thermo-mechanical and purely mechanical loads
        s_load = pym.MathGeneral("inp0 + inp1")(f, s_thermal_load)

        # Solve mechanical system of equations
        s_disp = pym.LinSolve()(s_K, s_load)
        s_disp.tag = "displacement"

        # Compliance
        s_compliance = pym.EinSum('i,i->')(s_disp, s_load)

        # Objective
        s_objective = pym.Scaling(scaling=10)(s_compliance)
        s_objective.tag = "Objective"

        # Output to Paraview VTI format
        pym.WriteToVTI(domain, saveto="out/dat.vti")(s_xfilt, s_T, s_disp)

        # Volume
        s_volume = pym.EinSum(expression='i->')(s_xfilt)

        # Volume constraint
        s_volume_constraint = pym.Scaling(scaling=1, maxval=volfrac * domain.nel)(s_volume)
        s_volume_constraint.tag = "Volume constraint"

        # Show iteration history
        pym.PlotIter()(s_objective, s_volume_constraint)

    # Optimization
    pym.minimize_mma(s_x, [s_objective, s_volume_constraint], verbosity=3, maxit=100)
