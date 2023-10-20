"""
Example of the design of cantilever for minimum volume subjected to displacement constraint.

References:
    None? s.koppen@tudelft.nl
"""

import numpy as np

import pymoto as pym
from pymoto.modules.assembly import get_B, get_D

# Problem settings
nx, ny = 100, 60  # Domain size
xmin, filter_radius, volfrac = 1e-6, 1.5, 1.0

scaling_objective = 10.0
maximum_vm_stress = 0.4


class StressConstraints(pym.Module):
    def _prepare(self, max_stress):
        self.max_stress = max_stress

    def _response(self, stress_vm):
        return (stress_vm / self.max_stress) - 1

    def _sensitivity(self, dfdv):
        return dfdv / self.max_stress


class ConstraintAggregation(pym.Module):

    def _prepare(self, P=10):
        self.P = P

    def _response(self, scaled_stress_constraints):
        self.g = scaled_stress_constraints + 1
        self.f = self.g ** self.P
        self.e = (1 / len(self.f)) * np.sum(self.f)
        d = self.e ** (1 / self.P)
        return d - 1

    def _sensitivity(self, dfdc):
        return (dfdc / len(self.f)) * self.e ** (1 / self.P - 1) * self.g ** (self.P - 1)


class Stress(pym.Module):
    def _prepare(self, E=1, nu=0.3, plane='strain', *args, **kwargs):
        siz = domain.element_size
        self.domain = domain

        # Constitutive model
        self.D = siz[2] * get_D(E, nu, plane.lower())

        # Vandermonde matrix
        self.V = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 3]])

        # Numerical integration
        self.B = np.zeros((3, 8), dtype=float)
        w = np.prod(siz[:domain.dim] / 2)
        for n in domain.node_numbering:
            pos = n * (siz / 2) / np.sqrt(3)  # Sampling point
            dN_dx = domain.eval_shape_fun_der(pos)
            self.B += w * get_B(dN_dx)

        self.dofconn = domain.get_dofconnectivity(2)

    def _response(self, u):
        self.elemental_strain = self.B.dot(u[self.dofconn].transpose())
        self.elemental_strain[2, :] *= 2  # voigt notation
        self.elemental_stress = self.D.dot(self.elemental_strain).transpose()
        self.stress_vm0 = (self.elemental_stress.dot(self.V) * self.elemental_stress).sum(1)
        return np.sqrt(self.stress_vm0)

    def _sensitivity(self, dfdv):
        sens = dfdv[:, np.newaxis] * (self.stress_vm0 ** (-0.5))[:, np.newaxis] * self.elemental_stress.dot(self.V)
        dgdsstrainmat = np.einsum('jk,kl->jl', sens, self.D)
        dgdsstrainmat[:, 2] *= 2
        dgdue = np.einsum('ij,jl->il', dgdsstrainmat, self.B)
        y = np.zeros(self.domain.nnodes * 2)
        for i in range(0, self.domain.nel):
            y[self.dofconn[i, :]] += dgdue[i, :]
        return y


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(nx, ny // 2) + 1] = 1.0

    # Initial design
    s_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    fn = pym.Network()

    # Filtering
    s_filtered_variables = fn.append(pym.DensityFilter(s_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    s_penalized_variables = fn.append(pym.MathGeneral(s_filtered_variables, expression=f"{xmin} + {1 - xmin}*inp0^3"))

    # Assemble stiffness matrix
    s_K = fn.append(pym.AssembleStiffness(s_penalized_variables, domain=domain, bc=dofs_left))

    # Solve
    s_force = pym.Signal('f', state=f)
    s_displacement = fn.append(pym.LinSolve([s_K, s_force]))

    # Calculate stress
    s_stress_vm = fn.append(Stress([s_displacement], domain=domain))

    s_stress_constraints = fn.append(StressConstraints([s_stress_vm], max_stress=maximum_vm_stress))
    s_stress_constraints_scaled = fn.append(
        pym.EinSum([s_filtered_variables, s_stress_constraints], expression='i,i->i'))
    s_stress_constraint = fn.append(ConstraintAggregation([s_stress_constraints_scaled], P=10))
    s_stress_constraint.tag = "Stress constraint"

    # Volume
    s_volume = fn.append(pym.EinSum(s_filtered_variables, expression='i->'))

    # Objective
    s_objective = fn.append(pym.Scaling([s_volume], scaling=scaling_objective))
    s_objective.tag = "Objective (volume)"

    # Plotting
    s_stress_scaled = fn.append(pym.EinSum([s_filtered_variables, s_stress_vm], expression='i,i->i'))
    module_plotstress = pym.PlotDomain(s_stress_scaled, domain=domain, cmap='jet')

    module_plotdomain = pym.PlotDomain(s_filtered_variables, domain=domain, saveto="out/design")
    responses = [s_objective, s_stress_constraint]
    module_plotiter = pym.PlotIter(responses)
    fn.append(module_plotdomain, module_plotstress, module_plotiter)

    # Optimization
    pym.minimize_mma(fn, [s_variables], responses, verbosity=2, maxit=300)
