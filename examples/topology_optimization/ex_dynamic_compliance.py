"""
Example of the design of cantilever for minimum dynamic compliance.

Implemented by @artofscience (s.koppen@tudelft.nl) based on:

Silva, O. M., Neves, M. M., & Lenzi, A. (2019).
A critical analysis of using the dynamic compliance as objective function
in topology optimization of one-material structures
considering steady-state forced vibration problems.
Journal of Sound and Vibration, 444, 1-20.
DOI: https://doi.org/10.1016/j.jsv.2018.12.030
"""
from math import pi

import numpy as np

import pymoto as pym

nx, ny = 100, 50

xmin, filter_radius, volfrac = 1e-9, 2, 0.5
E, rho = 1000, 1 / 1000

omega = 2.0  # excitation frequency (just below first eigenfrequency)
load = 1.0  # load magnitude

alpha, beta = 0.5, 0.5  # damping parameters

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
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(nx, ny // 2) + 1] = load

    # Initial design
    s_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    fn = pym.Network()

    # Filtering
    s_filtered_variables = fn.append(pym.DensityFilter(s_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    s_penalized_variables = fn.append(pym.MathGeneral(s_filtered_variables, expression=f"{xmin} + {1 - xmin}*inp0^3"))

    # Assemble stiffness matrix
    s_K = fn.append(pym.AssembleStiffness(s_penalized_variables, domain=domain, bc=dofs_left, e_modulus=E))

    # Assemble mass matrix
    s_M = fn.append(pym.AssembleMass(s_filtered_variables, domain=domain, bc=dofs_left, rho=rho))

    # Calculate the eigenfrequencies only once
    calculate_eigenfrequencies = True
    if calculate_eigenfrequencies:
        fn.response()
        m_eig = pym.EigenSolve([s_K, s_M], hermitian=True, nmodes=3)
        m_eig.response()
        eigfreq = np.sqrt(m_eig.sig_out[0].state)
        print(f"Eigenvalues are {eigfreq} rad/s or {eigfreq / (2 * pi)} Hz")

    # Build dynamic stiffness matrix Z
    s_omega = pym.Signal('omega', omega)
    s_Z = fn.append(DynamicMatrix([s_K, s_M, s_omega], alpha=alpha, beta=beta))

    # Solve
    s_force = pym.Signal('f', state=f)
    s_displacement = fn.append(pym.LinSolve([s_Z, s_force], pym.Signal('u')))

    # Output displacement (is a complex value)
    s_dynamic_compliance = fn.append(pym.EinSum([s_displacement, s_force], expression='i,i->'))

    # Absolute value (amplitude of response)
    s_dynamic_norm = fn.append(pym.ComplexNorm(s_dynamic_compliance))

    # Objective
    s_objective = fn.append(pym.Scaling([s_dynamic_norm], scaling=scaling_objective))
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
