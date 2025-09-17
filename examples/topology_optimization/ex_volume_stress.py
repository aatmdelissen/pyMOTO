"""Stress constraint
====================

Example of the design of cantilever for minimum volume subjected to stress constraints

It uses the following specific modules:

- :py:class:`pymoto.Stress` To calculate stresses from mechanical deformations
- :py:class:`pymoto.EinSum` Used to calculate von Mises stresses
- :py:class:`pymoto.Scaling` In this case used for scaling of all the stress constraints
- :py:class:`pymoto.PlotDomain` Next to the design itself, it can show the stress distribution

References:
  Verbart, A., Langelaar, M., & Keulen, F. V. (2017).
  A unified aggregation and relaxation approach for stress-constrained topology optimization.
  Structural and Multidisciplinary Optimization, 55, 663-679.
  DOI: https://doi.org/10.1007/s00158-016-1524-0
"""
import numpy as np
import pymoto as pym

# Problem settings
nx, ny = 50, 100  # Domain size
xmin, filter_radius, volfrac = 1e-9, 2, 1.0

maximum_vm_stress = 0.4  # Maximum allowed von-Mises stress

# Extra displacement constraint
displacement_constraint = True
max_displacement = 20.0 


class ConstraintAggregation(pym.Module):
    """Unified aggregation and relaxation.

    Implemented by @artofscience (s.koppen@tudelft.nl), based on:

    Verbart, A., Langelaar, M., & Keulen, F. V. (2017).
    A unified aggregation and relaxation approach for stress-constrained topology optimization.
    Structural and Multidisciplinary Optimization, 55, 663-679.
    DOI: https://doi.org/10.1007/s00158-016-1524-0
    """

    def __init__(self, P=10):
        self.P = P

    def __call__(self, x):
        """
        a = x + 1
        b = aggregation(a)
        c = b - 1
        """
        self.n = len(x)
        self.x = x
        self.y = self.x + 1
        self.z = self.y ** self.P
        z = (np.sum(self.z) / self.n) ** (1 / self.P)  # P-mean aggregation function
        return z - 1

    def _sensitivity(self, dfdc):
        return (dfdc / self.n) * (np.sum(self.z) / self.n) ** (1 / self.P - 1) * self.y ** (self.P - 1)


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.nodes[0, :]
    dofs_left = domain.get_dofnumber(nodes_left, [0, 1], ndof=2)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.nodes[nx, ny//2] + 1] = 1.0

    # Signal with design vector
    s_x = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    with pym.Network() as fn:
        # Density filter
        s_xfilt = pym.DensityFilter(domain, radius=filter_radius)(s_x)
        s_xfilt.tag = "Filtered density"

        # Plot the design
        pym.PlotDomain(domain, saveto="out/design")(s_xfilt)

        # SIMP penalization
        s_xsimp = pym.MathGeneral(f"{xmin} + {1 - xmin}*inp0^3")(s_xfilt)

        # Assembly of stiffness matrix
        s_K = pym.AssembleStiffness(domain, e_modulus=1.0, poisson_ratio=0.3, bc=dofs_left)(s_xsimp)

        # Solve
        s_displacement = pym.LinSolve()(s_K, f)

        # Calculate stress components
        s_stress = pym.Stress(domain)(s_displacement)

        # Calculate Von-Mises stress
        V = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 3]])  # Vandermonde matrix
        s_stress_vm2 = pym.EinSum('ij,ik,kj->j')(s_stress, V, s_stress)
        s_stress_vm = pym.MathGeneral('sqrt(inp0)')(s_stress_vm2)

        # Stress constraint
        s_stress_constraints = pym.Scaling(maxval=maximum_vm_stress, scaling=1.0)(s_stress_vm)
        s_stress_constraints_scaled = pym.EinSum('i,i->i')(s_xfilt, s_stress_constraints)
        s_stress_constraint = ConstraintAggregation(P=10)(s_stress_constraints_scaled)
        s_stress_constraint.tag = "Stress constraint"

        # Volume
        s_volume = pym.EinSum('i->')(s_xfilt)

        # Objective
        s_objective = pym.Scaling(scaling=100)(s_volume)
        s_objective.tag = "Objective (volume)"

        # Plot the stress values
        s_stress_scaled = pym.EinSum('i,i->i')(s_xfilt, s_stress_vm)
        s_stress_scaled.tag = 'Scaled von-Mises stress'
        pym.PlotDomain(domain, cmap='jet')(s_stress_scaled)

        # List of optimization responses: first the objective and all others the constraints
        responses = [s_objective, s_stress_constraint]

        if displacement_constraint:
            # Output displacement
            s_uout = pym.EinSum('i,i->')(s_displacement, f)

            # Displacement constraint
            s_uout_constraint = pym.Scaling(scaling=10, maxval=max_displacement)(s_uout)
            s_uout_constraint.tag = "Displacement constraint"

            responses.append(s_uout_constraint)

        pym.PlotIter()(*responses)

    # Optimization
    pym.minimize_mma(s_x, responses, verbosity=2, maxit=300)
