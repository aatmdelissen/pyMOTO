"""
Compliant mechanism
===================

Example of the design of a compliant mechanism using topology optimization

It considers the following conditions:

 1. maximum output displacement due to input load
 2. constrained minimum input and output stiffness

In this example of compliant mechanism design, no springs are required on input and output nodes. Also convergence is 
improved compared to `ex_compliant_mechanism_springs.py`. 

References:
  Koppen, S. (2022).
  Topology optimization of compliant mechanisms with multiple degrees of freedom.
  PhD thesis, Delft University of Technology.
  DOI: http://dx.doi.org/10.4233/uuid:21994a92-e365-4679-b6ac-11a2b70572b7
"""
import numpy as np
import pymoto as pym

# Problem settings
nx, ny = 60, 60  # Domain size
xmin, filter_radius, volfrac = 1e-6, 2, 0.3  # Density settings
nu, E = 0.3, 100  # Material properties

# Values on max compliance on input and output for constraints
max_input_compliance = 10.0
max_output_compliance = 10.0

use_volume_constraint = False

if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    nodes_right = domain.get_nodenumber(nx, np.arange(ny + 1))

    dofs_right = domain.get_dofnumber(nodes_right, [0, 1], ndof=2).flatten()
    dofs_left_x = 2*nodes_left
    dof_input = 2*nodes_left[0] + 1  # Input dof for mechanism
    dof_output = 2*nodes_left[-1] + 1  # Output dof for mechanism

    fixed_dofs = np.union1d(dofs_left_x, dofs_right)

    # Setup rhs for two loadcases 
    f = np.zeros((domain.nnodes * 2, 2), dtype=float)
    f[dof_output, 0] = 1.0
    f[dof_input, 1] = 1.0

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
        s_K = pym.AssembleStiffness(domain, e_modulus=E, poisson_ratio=nu, bc=fixed_dofs)(s_xsimp)

        # Solve for the displacements
        s_u = pym.LinSolve()(s_K, f)

        # Output displacement due to force on input
        s_uout = pym.EinSum('i,i->')(s_u[:, 1], f[:, 0])

        # Objective is the displacement at the output
        s_gobj = pym.Scaling(scaling=10.0)(s_uout)
        s_gobj.tag = "Objective"

        # Input and output compliances
        s_compl = pym.EinSum('ij,ij->j')(s_u, f)

        # Compliance constraint on input and output
        s_gcompl_out = pym.Scaling(scaling=10.0, maxval=max_output_compliance)(s_compl[0])
        s_gcompl_in = pym.Scaling(scaling=10.0, maxval=max_input_compliance)(s_compl[1])
        s_gcompl_out.tag = "Output compliance constraint"
        s_gcompl_in.tag = "Input compliance constraint"
    
        # List of optimization responses: first the objective and all others the constraints
        optimization_responses = [s_gobj, s_gcompl_out, s_gcompl_in]
        
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
