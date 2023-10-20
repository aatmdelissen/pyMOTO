"""
Example of the design of cantilever for minimum volume subjected to displacement constraint.

References:
    None? s.koppen@tudelft.nl
"""

import numpy as np

import pymoto as pym

# Problem settings
nx, ny = 40, 40  # Domain size
xmin, filter_radius, volfrac = 1e-9, 2, 1.0  # Density settings

scaling_objective = 10.0

scaling_volume_constraint = 10.0

load = -0.01  # point load
# gravity = -100.0 / (nx * ny)  # Gravity force
gravity = 0


class SelfWeight(pym.Module):
    def _prepare(self, gravity=1.0, domain=pym.DomainDefinition):
        self.gravity = gravity
        self.n = domain.nnodes * 2
        self.nel = domain.nel
        self.dofconn = domain.get_dofconnectivity(2)
        self.dfdx = np.zeros(self.nel, dtype=float)

    def _response(self, x, *args):
        f = np.zeros(self.n, dtype=float)
        np.add.at(f, self.dofconn[:, [0, 1, 3, 6]].flatten(), np.kron(x, -self.gravity * np.ones(4) / 4))
        np.add.at(f, self.dofconn[:, [2, 4, 5, 7]].flatten(), np.kron(x, self.gravity * np.ones(4) / 4))
        return f

    def _sensitivity(self, dfdv):
        dfdx = np.zeros(self.nel, dtype=float)
        for i in [0, 1, 3, 6]:
            dfdx[:] -= dfdv[self.dofconn[:, i]] * self.gravity * 2
        for i in [2, 4, 5, 7]:
            dfdx[:] += dfdv[self.dofconn[:, i]] * self.gravity * 2
        return dfdx


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))
    nodes_right = domain.get_nodenumber(nx, np.arange(ny + 1))

    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_left_x = dofs_left[0::2]
    dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_left_x, dofs_right]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(1, ny) + 1] = load

    # Initial design
    s_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

    # Setup optimization problem
    fn = pym.Network()

    # Filtering
    s_filtered_variables = fn.append(pym.DensityFilter(s_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    s_penalized_variables = fn.append(
        pym.MathGeneral(s_filtered_variables, expression=f"{xmin} + {1 - xmin}*(0.01*inp0 + 0.99*inp0^3)"))

    # Assemble stiffness matrix
    s_K = fn.append(pym.AssembleStiffness(s_penalized_variables, domain=domain, bc=None))

    # Solve
    s_force = pym.Signal('f', state=f)
    s_gravity = fn.append(SelfWeight([s_filtered_variables], gravity=gravity, domain=domain))

    s_load = fn.append(pym.MathGeneral([s_force, s_gravity], expression="inp0 + inp1"))

    s_up = pym.Signal('up', state=np.zeros(len(prescribed_dofs), dtype=float))
    s_state = fn.append(
        pym.SystemOfEquations([s_K, s_load[free_dofs], s_up], prescribed=prescribed_dofs, free=free_dofs))

    # Compliance
    s_compliance = fn.append(pym.EinSum([s_state[0], s_state[1]], expression='i,i->'))

    # Objective
    s_objective = fn.append(pym.Scaling([s_compliance], scaling=scaling_objective))
    s_objective.tag = "Objective"

    # Volume
    s_volume = fn.append(pym.EinSum(s_filtered_variables, expression='i->'))

    # Displacement constraint
    s_volume_constraint = fn.append(
        pym.Scaling(s_volume, scaling=scaling_volume_constraint, maxval=volfrac * domain.nel))
    s_volume_constraint.tag = "Volume constraint"

    # Plotting
    module_plotdomain = pym.PlotDomain(s_filtered_variables, domain=domain, saveto="out/design")
    responses = [s_objective, s_volume_constraint]
    module_plotiter = pym.PlotIter(responses)
    fn.append(module_plotdomain, module_plotiter)

    # Optimization
    pym.finite_difference(fn, [s_variables], responses)
