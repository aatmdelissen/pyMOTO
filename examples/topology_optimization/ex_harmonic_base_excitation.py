"""
Example of the design of cantilever beam with mass at tip, under harmonic base excitation.

Implemented by Derek Labaar and @artofscience (s.koppen@tudelft.nl) based on:

Zhu, J. H., He, F., Liu, T., Zhang, W. H., Liu, Q., & Yang, C. (2018).
Structural topology optimization under harmonic base acceleration excitations.
Structural and Multidisciplinary Optimization, 57, 1061-1078.
DOI: https://doi.org/10.1007/s00158-017-1795-0
"""

import numpy as np
from scipy.sparse import csc_matrix

import pymoto as pym

nx, ny = 80, 40
unitx, unity = 0.1, 0.1

xmin, filter_radius, volfrac = 1e-9, 2, 0.5
E, rho = 200e9, 7800

omega = 100  # excitation frequency (just below first eigenfrequency)
acceleration = 100.0  # load magnitude
mass = 1.0  # point mass magnitude

alpha, beta = 1e-5, 1e-7  # damping parameters

scaling_objective = 10.0
scaling_volume_constraint = 10.0


class DynamicMatrix(pym.Module):
    """ Constructs dynamic stiffness matrix with Rayleigh damping """

    def _prepare(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def _response(self, K, M, omega):
        return K + 1j * omega * (self.alpha * M + self.beta * K) - omega ** 2 * M

    def _sensitivity(self, dZ: pym.DyadCarrier):
        K, M, omega = [s.state for s in self.sig_in]
        dZr, dZi = dZ.real, dZ.imag
        dK = dZr - (omega * self.beta) * dZi
        dM = (-omega ** 2) * dZr - (omega * self.alpha) * dZi
        dZrM = dZr.contract(M)
        dZiK = dZi.contract(K)
        dZiM = dZi.contract(M)
        domega = -self.beta * dZiK - self.alpha * dZiM - 2 * omega * dZrM
        return dK, dM, domega


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny, unitx=unitx, unity=unity)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    prescribed_dofs = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_left_vertical = prescribed_dofs[1::2]

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    u = np.zeros_like(f)
    u[dofs_left_vertical] = acceleration

    # Setup partition of DOFs
    all_dofs = np.arange(0, 2 * domain.nnodes)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup optimization problem
    fn = pym.Network()

    # Initial design
    s_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Filtering
    s_filtered_variables = fn.append(pym.DensityFilter(s_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    s_penalized_variables = fn.append(
        pym.MathGeneral(s_filtered_variables, expression=f"{xmin} + {1 - xmin}*(0.01*inp0 + 0.99*inp0^3)"))

    # Assemble stiffness matrix
    s_K = fn.append(pym.AssembleStiffness(s_penalized_variables, domain=domain, e_modulus=E))

    # Point mass
    mass_node = 2 * domain.get_nodenumber(nx, ny // 2)
    mass_dofs = np.array([mass_node, mass_node + 1])
    mass_values = np.array([mass, mass])
    K_const = csc_matrix((mass_values, (mass_dofs, mass_dofs)), shape=(domain.nnodes * 2, domain.nnodes * 2))

    # Assemble mass matrix
    s_M = fn.append(pym.AssembleMass(s_filtered_variables, domain=domain, rho=rho, add_constant=K_const))

    # Build dynamic stiffness matrix Z
    s_omega = pym.Signal('omega', omega)
    s_Z = fn.append(DynamicMatrix([s_K, s_M, s_omega], alpha=alpha, beta=beta))

    # Solve
    s_f = pym.Signal('f', state=f[free_dofs])
    s_u = pym.Signal('u', state=u[prescribed_dofs])
    s_state = fn.append(pym.SystemOfEquations([s_Z, s_f, s_u], prescribed=prescribed_dofs))

    # Measure norm of deflection at point mass
    l = np.zeros_like(f)
    l[mass_node + 1] = 1.0
    s_l = pym.Signal('l', state=l)
    s_displacement = fn.append(pym.EinSum([s_state[0], s_l], expression='i,i->'))
    s_displacement_norm = fn.append(pym.ComplexNorm(s_displacement))

    # Objective
    s_objective = fn.append(pym.Scaling([s_displacement_norm], scaling=scaling_objective))
    s_objective.tag = "Objective"

    # Volume
    s_volume = fn.append(pym.EinSum(s_filtered_variables, expression='i->'))

    # Volume constraint
    s_volume_constraint = fn.append(
        pym.Scaling(s_volume, scaling=scaling_volume_constraint, maxval=volfrac * domain.nel))
    s_volume_constraint.tag = "Volume constraint"

    # Plotting
    module_plotdomain = pym.PlotDomain(s_filtered_variables, domain=domain, saveto="out/design")
    responses = [s_objective, s_volume_constraint]
    module_plotiter = pym.PlotIter(responses)
    fn.append(module_plotdomain, module_plotiter)

    # Optimization
    pym.minimize_mma(fn, [s_variables], responses, verbosity=2)
