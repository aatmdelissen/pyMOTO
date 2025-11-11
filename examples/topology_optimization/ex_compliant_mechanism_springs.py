"""Compliant mechanism (springs)
================================

Example showing the (in)efficacy of design of a compliant mechanism inverter using the traditional method of springs

The topology optimization considers:

1. Maximum output displacement due to input load
2. Constrained maximum volume
3. Spring attached to output
4. (Optional) spring attached to input

However, this problem formulation only works with

1. Spring with positive output stiffness
2. Non-design domains at the input and output
3. Active volume constraint
4. Input spring stiffness not equal to output spring stiffness

Even with these items taken into consideration, convergence is still troublesome as the inverter requires the 
displacement (initially positive) to pass trough zero in order to become negative. Other formulations offer better 
convergence properties, such as using mechanism modes and constraint modes as is done in 
:ref:`sphx_glr_auto_examples_topology_optimization_ex_compliant_mechanism_kinetostatic.py` or purely based on compliance
values in :ref:`sphx_glr_auto_examples_topology_optimization_ex_compliant_mechanism.py`.

The spring stiffnesses are added as a constant matrix using the :py:class:`pymoto.AssembleStiffness` module.

References:
  Bendsoe, M. P., & Sigmund, O. (2003).
  Topology optimization: theory, methods, and applications.
  Springer Science & Business Media.
  DOI: https://doi.org/10.1007/978-3-662-05086-6
"""
import numpy as np
from scipy.sparse import csc_matrix
import pymoto as pym

# Problem settings
nx, ny = 80, 80  # Domain size
xmin, filter_radius, volfrac = 1e-6, 2, 0.3  # Density settings
nu, E = 0.3, 100  # Material properties

# Spring stiffnesses for the virtual springs at the input and output
input_spring_stiffness = 5.0
output_spring_stiffness = 15.0

nondesigndomain_size = 4  # Size of the non-design domain at the input and output

use_volume_constraint = True

if __name__ == "__main__":
    # Set up the domain
    domain = pym.VoxelDomain(nx, ny)

    # Node and dof groups
    nodes_left = domain.nodes[0, :].flatten()
    nodes_right = domain.nodes[-1, :].flatten()

    dofs_right = domain.get_dofnumber(nodes_right, [0, 1], ndof=2).flatten()
    dofs_left_x = 2*nodes_left
    dof_input = 2*nodes_left[0] + 1  # Input dof for mechanism
    dof_output = 2*nodes_left[-1] + 1  # Output dof for mechanism

    fixed_dofs = np.union1d(dofs_left_x, dofs_right)

    # Setup rhs for input force
    f = np.zeros(domain.nnodes * 2, dtype=float)
    f[dof_input] = 1.0

    # Signal with design variables
    s_x = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    with pym.Network() as fn:
        # Setup non-design domains
        input_domain = domain.elements[:nondesigndomain_size, :nondesigndomain_size]
        output_domain = domain.elements[:nondesigndomain_size, -nondesigndomain_size:]
        non_design_domain = np.union1d(input_domain, output_domain)
        s_xnondes = pym.SetValue(indices=non_design_domain, value=1.0)(s_x)

        # Density filtering
        s_xfilt = pym.DensityFilter(domain, radius=filter_radius)(s_xnondes)
        s_xfilt.tag = 'Filtered density'

        # Plot the design
        pym.PlotDomain(domain, saveto="out/design")(s_xfilt)

        # SIMP penalization
        s_xsimp = pym.MathExpression(f"{xmin} + {1 - xmin}*inp0^3")(s_xfilt)

        # Assembly (with added constant stiffness for springs)
        istiff = np.array([dof_input, dof_output])
        sstiff = np.array([input_spring_stiffness, output_spring_stiffness])
        K_const = csc_matrix((sstiff, (istiff, istiff)), shape=(domain.nnodes*2, domain.nnodes*2))
        s_K = pym.AssembleStiffness(domain, e_modulus=E, poisson_ratio=nu, bc=fixed_dofs, add_constant=K_const)(s_xsimp)

        # Solve for the displacements
        s_u = pym.LinSolve()(s_K, f)

        # Objective is the displacement at the output
        s_gobj = pym.Scaling(scaling=10.0)(s_u[dof_output])
        s_gobj.tag = "Objective"

        # List of optimization responses: first the objective and all others the constraints
        optimization_responses = [s_gobj]

        # Add volume constraint if requested
        if use_volume_constraint:
            # Volume
            s_vol = pym.EinSum('i->')(s_xfilt)
            s_vol.tag = "Volume"

            # Volume constraint
            s_gvol = pym.Scaling(scaling=10.0, maxval=volfrac * domain.nel)(s_vol)
            s_gvol.tag = "Volume constraint"

            optimization_responses.append(s_gvol)

        # Plot iteration history
        pym.PlotIter()(*optimization_responses)

    # Optimization
    pym.minimize_mma(s_x, optimization_responses, verbosity=2)
