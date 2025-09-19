"""Self-weight gravity
======================

Example of the design of an arch; compliance minimization under self-weight.

Note: This is a difficult problem to optimize (in terms of convergence), because the load is dependent on the design. 

With modifications on:
(i) interpolation of Young's modulus (SIMPLIN)

Consequently, in contrast to Bruyneel et al. (2005), no special treatment of the sequential approximate optimization 
algorithm is required.

References:
  Bruyneel, M., & Duysinx, P. (2005).
  Note on topology optimization of continuum structures including self-weight.
  Structural and Multidisciplinary Optimization, 29, 245-256.
  DOI: https://doi.org/10.1007/s00158-004-0484-y

  Zhu, J., Zhang, W., & Beckers, P. (2009). 
  Integrated layout design of multi-component system. 
  International Journal for Numerical Methods in Engineering, 78(6), 631-651. 
  DOI: https://doi.org/10.1002/nme.2499
        
"""

import numpy as np
import pymoto as pym

# Problem settings
nx, ny = 100, 50  # Domain size
xmin, filter_radius = 1e-6, 1.5
initial_volfrac = 1.0

# Material properties (Aluminium)
E = 70e9  # Young's modulus (Pa)
nu = 0.3
rho = 2700  # kg/m^3

# Loadcase
load = 0.0 # 1000.0  # Constant point load on the top left of the domain, in negative y direction (N)
gravity = np.array([0.0, -1.0]) * 9.81  # Gravity (m/s^2)

""" Choose boundary condition type
1: arch
2: mbb-beam
3: fully-clamped
4: double-arch
"""
bc = 1

# Constraint on maximum volume
use_volume_constraint = False
max_volume = 0.2  # Maximum allowed volume fraction

# Constraint on maximum stress
use_stress_constraint = False
maximum_vm_stress = 0.2e+6  # Maximum stress value (Pa)


if __name__ == "__main__":
    # Set up the domain of size (1, 1*ny/nx, 0.1)m
    domain = pym.DomainDefinition(nx, ny, unitx=1/nx, unity=1/nx, unitz=0.1)
    
    # Node and dof groups for boundary conditions
    if bc == 1:  # Arch
        nodes_right = domain.nodes[-6:-1, 0].flatten()
        dofs_right = domain.get_dofnumber(nodes_right, [0, 1], 2)
    elif bc == 2:  # MBB-beam
        nodes_right = domain.nodes[-6:-1, 0].flatten()
        dofs_right = domain.get_dofnumber(nodes_right, 1, 2)
    elif bc == 3:  # Fully clamped
        nodes_right = domain.nodes[-1, ...].flatten()
        dofs_right = domain.get_dofnumber(nodes_right, [0, 1], 2)
    elif bc == 4:  # Double arch
        nodes_right = domain.nodes[-1, ...].flatten()
        dofs_right = domain.get_dofnumber(nodes_right, 1, 2)
    else:
        raise NotImplementedError("'bc' must be 1, 2, 3, or 4")

    # Symmetric bc on the left (rollers)
    nodes_left = domain.nodes[0, ...]
    dofs_left_x = nodes_left*2

    all_dofs = np.arange(0, 2 * domain.nnodes)
    fixed_dofs = np.unique(np.hstack([dofs_left_x.flatten(), dofs_right.flatten()]))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Setup rhs for constant load contribution
    f_const = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f_const[2 * domain.nodes[0:5, -1] + 1] = -load/5

    # Design variables
    s_variables = pym.Signal('x', state=initial_volfrac * np.ones(domain.nel))

    # Setup optimization problem
    with pym.Network() as fn:
        # Density filtering
        s_filtered_variables = pym.DensityFilter(domain, radius=filter_radius)(s_variables)
        s_filtered_variables.tag = "Filtered design"

        # SIMP penalization
        """
        Note the use of SIMPLIN: y = xmin + (1-xmin) * (alpha * x + (alpha - 1) * x^p)

        References:

        Zhu, J., Zhang, W., & Beckers, P. (2009). Integrated layout design of multi-component system. 
        International Journal for Numerical Methods in Engineering, 78(6), 631-651. https://doi.org/10.1002/nme.2499
        """
        s_penalized_variables = pym.MathGeneral(f"{xmin} + {1 - xmin}*(0.01*inp0 + 0.99*inp0^3)")(s_filtered_variables)

        # Assemble stiffness matrix
        s_K = pym.AssembleStiffness(domain, e_modulus=E, poisson_ratio=nu, bc=fixed_dofs)(s_penalized_variables)

        # Determine gravity load
        mass_el = rho * np.prod(domain.element_size)  # Mass of one element
        s_mass = pym.MathGeneral(f"{mass_el} * inp0")(s_filtered_variables)  # Scaled element mass
        s_gravity = pym.NodalOperation(domain, np.repeat(gravity/4, 4))(s_mass)

        # Add gravity load and constant load together
        s_load0 = pym.MathGeneral("inp0 + inp1")(f_const, s_gravity)

        # Set the load on the boundary condition to zero
        s_load = pym.VecSet(indices=fixed_dofs, value=0.0)(s_load0)

        # Solve linear system of equations
        s_u = pym.LinSolve()(s_K, s_load)

        # Compliance
        s_compliance = pym.EinSum('i,i->')(s_u, s_load)

        # Objective
        s_objective = pym.Scaling(scaling=1000)(s_compliance)
        s_objective.tag = "Objective"

        responses = [s_objective]

        # Plotting
        pym.PlotDomain(domain, saveto="out/design")(s_filtered_variables)

        if use_stress_constraint:
            # Calculate stress
            s_stress = pym.Stress(domain, e_modulus=E, poisson_ratio=nu)(s_u)
            V = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 3]])  # Vandermonde matrix
            s_stress_vm2 = pym.EinSum("ij,ik,kj->j")(s_stress, V, s_stress)
            s_stress_vm = pym.MathGeneral('sqrt(inp0)*inp1')(s_stress_vm2, s_penalized_variables)

            # Approximate maximum stress with aggregation
            s_stress_agg = pym.PNorm(p=2, 
                                     scaling=pym.AggScaling('max', 0.3), 
                                     active_set=pym.AggActiveSet(lower_rel=0.5))(s_stress_vm)
            s_stress_constraint = pym.Scaling(scaling=10, maxval=maximum_vm_stress)(s_stress_agg)
            s_stress_constraint.tag = "Stress constraint"
            responses.append(s_stress_constraint)

            # Plotting
            s_stress_scaled = pym.MathGeneral("inp0/1e+6")(s_stress_vm)
            s_stress_scaled.tag = "Von-Mises stress (MPa)"
            pym.PlotDomain(domain, cmap='jet')(s_stress_scaled)

        if use_volume_constraint:
            # Volume
            s_total_volume = pym.EinSum("i->")(s_filtered_variables)
            
            # Volume constraint
            s_volume_constraint = pym.Scaling(scaling=10, maxval=max_volume*domain.nel)(s_total_volume)
            s_volume_constraint.tag = "Volume constraint"
            responses.append(s_volume_constraint)
    
        pym.PlotIter()(*responses)

    # pym.finite_difference(s_variables, responses, dx=1e-4)

    # Optimization (try different algorithms by uncommenting)
    # pym.minimize_slp(s_variables, responses)  # Only unconstrained
    # pym.minimize_oc(s_variables, responses[0], verbosity=2, maxit=300)  # Only unconstrained
    pym.minimize_mma(s_variables, responses, verbosity=2, maxit=300, move=0.1)
