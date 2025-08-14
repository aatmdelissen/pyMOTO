"""Compliant mechanism (static condensation)
============================================

Usage of static condensation to design a compliant mechanism

This example uses :py:class:`pymoto.StaticCondensation` to calculate the mechanism and constraint mode compliances in an 
efficient way. The optimization problem considers

1. Maximum stiffness of input and output ports
2. Desired geometric advantage specified by the user
3. Desired maximum stiffness of the compliant deformation pattern

References:
  Koppen, S. (2022).
  Topology optimization of compliant mechanisms with multiple degrees of freedom.
  PhD thesis, Delft University of Technology.
  DOI: http://dx.doi.org/10.4233/uuid:21994a92-e365-4679-b6ac-11a2b70572b7
"""
import numpy as np
import pymoto as pym

# Problem settings
nx, ny = 100, 100  # Domain size
xmin, filter_radius, volfrac = 1e-6, 2, 0.35  # Density settings
nu, E = 0.3, 100  # Material properties

u_desired = np.array([1.0, -1.1])  # Intended geometric advantage of the mechanism [u_in, u_out]

compliance_constraint_value = 0.1  # Maximum compliance of the requested mechanism mode
use_volume_constraint = True

if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.nodes[0, :].flatten()
    nodes_right = domain.nodes[-1, :].flatten()

    dofs_right = domain.get_dofnumber(nodes_right, [0, 1], ndof=2).flatten()
    dofs_left_x = 2*nodes_left

    dof_input = 2*nodes_left[0] + 1  # Input dof for mechanism
    dof_output = 2*nodes_left[-1] + 1  # Output dof for mechanism

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.array(list({*dofs_left_x, *dofs_right})) 
    io_dofs = np.array([dof_input, dof_output])
    free_dofs = np.setdiff1d(all_dofs, np.concatenate([io_dofs, prescribed_dofs]))

    # Signal with design variables
    s_x = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    with pym.Network() as fn:
        # Density filtering
        s_xfilt = pym.DensityFilter(domain, radius=filter_radius)(s_x)
        s_xfilt.tag = 'Filtered density'

        # Plot the design
        pym.PlotDomain(domain, saveto="out/design")(s_xfilt)

        # SIMP penalization
        s_xsimp = pym.MathGeneral(f"{xmin} + {1 - xmin}*inp0^3")(s_xfilt)

        # Assembly of stiffness matrix
        s_K = pym.AssembleStiffness(domain, e_modulus=E, poisson_ratio=nu)(s_xsimp)

        # Static condensation of the dofs of interest
        s_Kr = pym.StaticCondensation(main=io_dofs, free=free_dofs)(s_K)

        # Compliance of the mechanism mode
        s_cmech = pym.EinSum('i,ik,k->')(u_desired, s_Kr, u_desired)

        # Compliance constraint on the mechanism mode
        s_gcmech = pym.Scaling(scaling=10.0, maxval=compliance_constraint_value)(s_cmech)
        s_gcmech.tag = "Mechanism compliance"

        # Inverse of the reduced stiffness matrix (=compliance matrix)
        s_Y = pym.Inverse()(s_Kr)

        # Sum the diagonal only (input and output compliance); off-diagonal terms are input-output compliance
        s_c = pym.EinSum('ii->')(s_Y)

        # Objective function
        s_gobj = pym.Scaling(scaling=100.0)(s_c)
        s_gobj.tag = "Objective"

        # List of optimization responses: first the objective and all others the constraints
        optimization_responses = [s_gobj, s_gcmech]

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
