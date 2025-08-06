"""
Compliance with prescribed displacements
========================================

Example of the design for stiffness of a structure subjected to prescribed displacements
 using topology optimization with:
(i) maximum stiffness between prescribed displacements and support, and
(ii) constrained by maximum volume.

Reference:

Koppen, S., Langelaar, M., & van Keulen, F. (2022). 
A simple and versatile topology optimization formulation for flexure synthesis. 
Mechanism and Machine Theory, 172, 104743.
DOI: http://dx.doi.org/10.1016/j.mechmachtheory.2022.104743
"""

import numpy as np
import pymoto as pym

# Problem settings
nx, ny = 40, 40  # Domain size
xmin, filter_radius, volfrac = 1e-9, 2, 0.5  # Density settings
nu, E = 0.3, 1.0  # Material properties

if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.nodes[0, :].flatten()
    nodes_right = domain.nodes[-1, :].flatten()
    dofs_left_x = domain.get_dofnumber(nodes_left, 0, 2).flatten()
    dofs_left_y = domain.get_dofnumber(nodes_left, 1, 2).flatten()
    dofs_right = domain.get_dofnumber(nodes_right, np.arange(2), 2).flatten()

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.concatenate([dofs_left_x, dofs_right, dofs_left_y]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup solution vectors and rhs
    ff = np.zeros_like(free_dofs, dtype=float)
    u = np.zeros_like(all_dofs, dtype=float)

    u[dofs_left_y] = 1.0
    up = u[prescribed_dofs]

    # Initial design
    s_x = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    with pym.Network() as fn:
        # Density filtering
        s_xfilt = pym.DensityFilter(domain, radius=filter_radius)(s_x)
        s_xfilt.tag = 'xfiltered'

        # SIMP penalization
        s_xsimp = pym.MathGeneral(f"{xmin} + {1 - xmin}*inp0^3")(s_xfilt)

        # Matrix assembly
        s_K = pym.AssembleStiffness(domain, e_modulus=E, poisson_ratio=nu)(s_xsimp)

        # Solve system of equations
        s_u, s_f = pym.SystemOfEquations(prescribed=prescribed_dofs)(s_K, ff, up)

        # Calculate compliance value
        s_c = pym.EinSum('i,i->')(s_u, s_f)

        # Objective function
        s_gobj =pym.Scaling(scaling=-1)(s_c)
        s_gobj.tag = "Objective"

        # Volume
        s_vol = pym.EinSum('i->')(s_xfilt)
        s_vol.tag = "volume"

        # Volume constraint
        s_gvol = pym.Scaling(scaling=10.0, maxval=volfrac * domain.nel)(s_vol)
        s_gvol.tag = "Volume constraint"

        # Plotting
        pym.PlotDomain(domain, saveto="out/design")(s_xfilt)
        optimization_responses = [s_gobj, s_gvol]
        pym.PlotIter()(*optimization_responses)

    # Run optimization
    pym.minimize_mma(s_x, optimization_responses, verbosity=2)
