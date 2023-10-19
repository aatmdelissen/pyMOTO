"""
Example of the design of cantilever for minimum dynamic compliance.
"""
from math import pi

import numpy as np

# flake8: noqa
import pymoto as pym

# Problem settings
lx, ly = 1, 0.5
nx, ny = int(lx * 100), int(ly * 100)  # Domain size
unitx, unity = lx / nx, ly / ny
unitz = 1.0  # ?

xmin, filter_radius, volfrac = 1e-6, 2, 0.49  # Density settings
nu = 0.3  # Material properties

E = 210e9  # 210 GPa (Silva, 2018)
rho = 7860  # kg/m3 (Silva, 2018)

# fundamental eigenfreq should be around 175 Hz (Silva, 2018)
omega = 180 * (2 * pi)

# force
force_magnitude = -9000  # (Silva, 2018)

# damping parameters
alpha = 1e-3
beta = 1e-8

scaling_objective = 10.0
scaling_volume_constraint = 10.0


class DynamicMatrix(pym.Module):
    def _prepare(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def _response(self, K, M, omega):
        return K + 1j * omega * (self.alpha * M + self.beta * K) - omega ** 2 * M

    def _sensitivity(self, dZ):
        K, M, omega = [s.state for s in self.sig_in]
        dK = np.real(dZ) - (omega * self.beta) * np.imag(dZ)
        dM = (-omega ** 2) * np.real(dZ) - (omega * self.alpha) * np.imag(dZ)
        dZrM = np.real(dZ).contract(M)
        dZiK = np.imag(dZ).contract(K)
        dZiM = np.imag(dZ).contract(M)
        domega = -self.beta * dZiK - self.alpha * dZiM - 2 * omega * dZrM
        return dK, dM, domega


class ComplexVecDot(pym.Module):
    def _response(self, u, v):
        return u @ v

    def _sensitivity(self, dy):
        u, v = [s.state for s in self.sig_in]
        return dy * v, dy * u


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny, unitx=unitx, unity=unity, unitz=unitz)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(nx, ny // 2) + 1] = force_magnitude

    # Initial design
    signal_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    network = pym.Network()

    # Filtering
    signal_filtered_variables = network.append(pym.DensityFilter(signal_variables, domain=domain, radius=filter_radius))

    # Penalization
    signal_penalized_variables = network.append(
        pym.MathGeneral(signal_filtered_variables, expression=f"{xmin} + {1 - xmin}*inp0^3"))

    # Assemble stiffness matrix
    signal_stiffness = network.append(
        pym.AssembleStiffness(signal_penalized_variables, domain=domain, e_modulus=E, poisson_ratio=nu, bc=dofs_left))

    # Assemble mass matrix
    signal_mass = network.append(pym.AssembleMass(signal_penalized_variables, domain=domain, bc=dofs_left))

    # Build dynamic stiffness matrix
    signal_omega = pym.Signal('omega', omega)
    signal_dynamic_stiffness = network.append(
        DynamicMatrix([signal_stiffness, signal_mass, signal_omega], alpha=alpha, beta=beta))

    # Solve
    signal_force = pym.Signal('f', state=f)
    signal_displacement = network.append(pym.LinSolve([signal_dynamic_stiffness, signal_force], pym.Signal('u')))

    # Output displacement
    signal_dynamic_compliance = network.append(ComplexVecDot([signal_displacement, signal_force]))

    # Objective
    signal_objective = network.append(pym.Scaling([signal_dynamic_compliance], scaling=scaling_objective))
    signal_objective.tag = "Objective"

    # Volume
    signal_volume = network.append(pym.EinSum(signal_filtered_variables, expression='i->'))
    signal_volume.tag = "volume"

    # Volume constraint
    signal_volume_constraint = network.append(
        pym.Scaling(signal_volume, scaling=scaling_volume_constraint, maxval=volfrac * domain.nel))
    signal_volume_constraint.tag = "Volume constraint"

    # Plotting
    module_plotdomain = pym.PlotDomain(signal_filtered_variables, domain=domain, saveto="out/design")
    responses = [signal_objective, signal_volume_constraint]
    module_plotiter = pym.PlotIter(responses)
    network.append(module_plotdomain, module_plotiter)

    # Optimization
    pym.minimize_mma(network, [signal_variables], responses, verbosity=2)
