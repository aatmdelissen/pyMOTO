"""
Example of the design of a thermoelastic structure.

Implementation by @artofscience (s.koppen@tudelft.nl) based on:

Gao, T., & Zhang, W. (2010).
Topology optimization involving thermo-elastic stress loads.
Structural and multidisciplinary optimization, 42, 725-738.
DOI: https://doi.org/10.1007/s00158-010-0527-5
"""

import numpy as np

import pymoto as pym

# Problem settings
nx, ny = 60, 80  # Domain size
xmin, filter_radius = 1e-9, 2

load = -100.0  # point load

scaling_objective = 10.0

volfrac = 0.25
scaling_volume_constraint = 1.0


class ThermalExpansionLoading(pym.Module):
    def _prepare(self, alpha=10, domain=pym.DomainDefinition):
        self.alpha = alpha
        self.dofconn = domain.get_dofconnectivity(2)
        self.f = np.zeros(domain.nnodes * 2, dtype=float)
        self.dfdx = np.zeros(domain.nel, dtype=float)

    def _response(self, x, *args):
        self.f[:] = 0.0
        np.add.at(self.f, self.dofconn[:, [0, 1, 3, 4]].flatten(), -self.alpha * np.kron(x, np.ones(4)) / 4)
        np.add.at(self.f, self.dofconn[:, [2, 5, 6, 7]].flatten(), self.alpha * np.kron(x, np.ones(4)) / 4)
        return self.f

    def _sensitivity(self, dfdv):
        self.dfdx[:] = 0.0
        self.dfdx[:] -= dfdv[self.dofconn[:, [0, 1, 3, 4]]].sum(1) * self.alpha / 4
        self.dfdx[:] += dfdv[self.dofconn[:, [2, 5, 6, 7]]].sum(1) * self.alpha / 4
        return self.dfdx


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    nodes_right = domain.get_nodenumber(nx, np.arange(ny + 1))
    dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))

    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_left_x = dofs_left[0::2]

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_left_x, dofs_right]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(1, 0) + 1] = load

    # Initial design
    s_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    fn = pym.Network()

    # Filtering
    s_filtered_variables = fn.append(pym.DensityFilter(s_variables, domain=domain, radius=filter_radius))

    # RAMP with q = 1
    s_penalized_variables = fn.append(
        pym.MathGeneral(s_filtered_variables, expression=f"{xmin} + {1 - xmin}*(inp0 / (2 - inp0))"))

    # Assemble stiffness matrix
    s_K = fn.append(pym.AssembleStiffness(s_penalized_variables, domain=domain, bc=None))

    # Solve
    s_force = pym.Signal('f', state=f)
    s_gravity = fn.append(ThermalExpansionLoading([s_filtered_variables], domain=domain))

    s_load = fn.append(pym.MathGeneral([s_force, s_gravity], expression="inp0 + inp1"))

    s_up = pym.Signal('up', state=np.zeros(len(prescribed_dofs), dtype=float))
    s_state = fn.append(
        pym.SystemOfEquations([s_K, s_load[free_dofs], s_up], prescribed=prescribed_dofs, free=free_dofs))

    # Compliance
    s_compliance = fn.append(pym.EinSum([s_state[0], s_state[1]], expression='i,i->'))

    # Objective
    s_objective = fn.append(pym.Scaling([s_compliance], scaling=scaling_objective))
    s_objective.tag = "Objective"

    # Plotting
    fn.append(pym.PlotDomain(s_filtered_variables, domain=domain, saveto="out/design"))
    responses = [s_objective]
    plot_signals = responses.copy()

    # Volume
    s_volume = fn.append(pym.EinSum(s_filtered_variables, expression='i->'))

    # Volume constraint
    s_volume_constraint = fn.append(
        pym.Scaling(s_volume, scaling=scaling_volume_constraint, maxval=volfrac * domain.nel))
    s_volume_constraint.tag = "Volume constraint"
    responses.append(s_volume_constraint)
    plot_signals.append(s_volume_constraint)

    fn.append(pym.PlotIter(plot_signals))

    # Optimization
    pym.minimize_mma(fn, [s_variables], responses, verbosity=2, maxit=300)
