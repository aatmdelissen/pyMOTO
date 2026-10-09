"""Multi-material thermo-mechanical
===================================

Example of the design of a thermoelastic structure with combined heat and mechanical load, using multiple materials

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

  Sigmund, O. (2001).
  Design of multiphysics actuators using topology optimization, part II.
  Computer methods in applied mechanics and engineering, 190(49-50).
  DOI: https://doi.org/10.1016/S0045-7825(01)00252-3
"""
import numpy as np
import pymoto as pym

# Problem settings
nx, ny = 60, 80  # Domain size
xmin, filter_radius = 1e-9, 2

load = -1.0  # Mechanical force
heatload = 1000000.0  # Thermal heat load

volfrac = 0.25

# Material definition
mat = [None for _ in range(3)]
mat[0] = dict(E=1.0000, nu=0.00, k=1.e-6, cte=0.0)  # Void material
mat[1] = dict(E=200e+9, nu=0.30, k=160.0, cte=150e-6)  # Steel
mat[2] = dict(E=110e+9, nu=0.34, k=400.0, cte=16.4e-6)  # Copper


if __name__ == "__main__":
    # Set up the domain
    domain = pym.VoxelDomain(nx, ny, unitx=1/nx, unity=1/nx)

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
    s_x1 = pym.Signal('x1', state=0.5 * np.ones(domain.nel))
    s_x2 = pym.Signal('x2', state=0.5 * np.ones(domain.nel))

    # Compute stiffness matrices
    dummy_domain = pym.VoxelDomain(1, 1, unitx=domain.unitx, unity=domain.unity, unitz=domain.unitz)
    for m in mat:
        m['Kel'] = pym.AssembleStiffness(dummy_domain, e_modulus=m['E'], poisson_ratio=m['nu']).elmat[0]
        m['Tel'] = pym.AssemblePoisson(dummy_domain, material_property=m['k']).elmat[0]

    # Setup optimization problem
    with pym.Network() as fn:
        # Density filtering
        s_x0filt = pym.DensityFilter(domain, radius=filter_radius)(s_x1)
        s_x0filt.tag = 'X1 Filtered'
        s_x1filt = pym.DensityFilter(domain, radius=filter_radius)(s_x2)
        s_x1filt.tag = 'X2 Filtered'

        s_mat0 = pym.MathExpression("1 - inp0")(s_x0filt)  # Contribution of void
        s_mat1 = pym.MathExpression("inp0 * inp1")(s_x0filt, s_x1filt)
        s_mat2 = pym.MathExpression("inp0 * (1-inp1)")(s_x0filt, s_x1filt)
        s_mat0.tag = 'void'
        s_mat1.tag = 'steel'
        s_mat2.tag = 'copper'

        # Plot the design
        pym.PlotDomain(domain, saveto="out/design")(s_mat1)
        pym.PlotDomain(domain, saveto="out/design")(s_mat2)

        # RAMP with q = 2
        s_xRAMP0 = pym.MathExpression("inp0")(s_mat0)
        s_xRAMP1 = pym.MathExpression("inp0 / (3 - 2*inp0)")(s_mat1)
        s_xRAMP2 = pym.MathExpression("inp0 / (3 - 2*inp0)")(s_mat2)

        # Assemble stiffness and conductivity matrix
        s_K = pym.AssembleGeneral(domain, bc=fixed_dofs, element_matrix=[m['Kel'] for m in mat])(s_xRAMP0, s_xRAMP1, s_xRAMP2)
        s_KT = pym.AssembleGeneral(domain, bc=nodes_right, element_matrix=[m['Tel'] for m in mat])(s_xRAMP0, s_xRAMP1, s_xRAMP2)

        # Solve for temperature
        s_T = pym.LinSolve()(s_KT, q)
        s_T.tag = "temperature"

        # Determine thermo-mechanical load
        s_Telem = pym.ElementAverage(domain)(s_T)
        s_xT1 = pym.MathExpression("inp0 * inp1")(s_Telem, s_mat1)
        s_thermal_load1 = pym.ThermoMechanical(domain, alpha=mat[1]['cte'])(s_xT1)
        s_xT2 = pym.MathExpression("inp0 * inp1")(s_Telem, s_mat2)
        s_thermal_load2 = pym.ThermoMechanical(domain, alpha=mat[2]['cte'])(s_xT2)

        # Combine thermo-mechanical and purely mechanical loads
        s_load = pym.MathExpression("inp0 + inp1 + inp2")(f, s_thermal_load1, s_thermal_load2)

        # Solve mechanical system of equations
        s_disp = pym.LinSolve()(s_K, s_load)
        s_disp.tag = "displacement"

        # Compliance
        s_compliance = pym.EinSum('i,i->')(s_disp, s_load)

        # Objective
        s_objective = pym.Scaling(scaling=10)(s_compliance)
        s_objective.tag = "Objective"

        # Output to Paraview VTI format
        pym.WriteToVTI(domain, saveto="out/dat.vti")(s_mat0, s_mat1, s_mat2, s_T, s_disp)

        # Volume
        s_mat0a = pym.MathExpression("1 - inp0")(s_mat0) 
        s_volume = pym.EinSum(expression='i->')(s_mat0a)

        # Volume constraint
        s_volume_constraint = pym.Scaling(scaling=1, maxval=volfrac * domain.nel)(s_volume)
        s_volume_constraint.tag = "Volume constraint"

        # Show iteration history
        pym.PlotIter()(s_objective, s_volume_constraint)

    # Optimization
    pym.minimize_mma([s_x1, s_x2], [s_objective, s_volume_constraint], verbosity=3, maxit=100)
