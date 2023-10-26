"""
Example of the design of an arch; compliance minimization under self-weight.

Implemented by @artofscience (s.koppen@tudelft.nl) based on:

Bruyneel, M., & Duysinx, P. (2005).
Note on topology optimization of continuum structures including self-weight.
Structural and Multidisciplinary Optimization, 29, 245-256.

With modifications on:
(i) interpolation of Young's modulus (SIMPLIN)

Consequencly, in contrast to Bruyneel et al. (2005),
no special treatment of the sequential approximate optimization algorithm is required.
"""

import numpy as np

import pymoto as pym

# Problem settings
nx, ny = 100, 50  # Domain size
xmin, filter_radius = 1e-9, 2
initial_volfrac = 1.0

load = 0.0  # point load
gravity = np.array([0.0, -1.0]) / (nx * ny)  # Gravity force

bc = 2

"""
1: arch
2: mbb-beam
3: fully-clamped
4: double-arch
"""

scaling_objective = 100.0

use_volume_constraint = False
volfrac = 0.2
scaling_volume_constraint = 10.0


class SelfWeight(pym.Module):
    def _prepare(self, gravity=np.array([0.0, -1.0], dtype=float), domain=pym.DomainDefinition):
        self.load_x = gravity[0] / 4
        self.load_y = gravity[1] / 4
        self.dofconn = domain.get_dofconnectivity(2)
        self.f = np.zeros(domain.nnodes * 2, dtype=float)
        self.dfdx = np.zeros(domain.nel, dtype=float)

    def _response(self, x, *args):
        self.f[:] = 0.0
        load_x = np.kron(x, self.load_x * np.ones(4))
        load_y = np.kron(x, self.load_y * np.ones(4))
        np.add.at(self.f, self.dofconn[:, 0::2].flatten(), load_x)
        np.add.at(self.f, self.dofconn[:, 1::2].flatten(), load_y)
        return self.f

    def _sensitivity(self, dfdv):
        self.dfdx[:] = 0.0
        self.dfdx[:] += dfdv[self.dofconn[:, 0::2]].sum(1) * self.load_x
        self.dfdx[:] += dfdv[self.dofconn[:, 1::2]].sum(1) * self.load_y
        return self.dfdx


if __name__ == "__main__":
    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    if bc == 1:
        nodes_right = domain.get_nodenumber(nx, np.arange(1))
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), 1)
    elif bc == 2:
        nodes_right = domain.get_nodenumber(nx, np.arange(1))
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), 1)[1]
    elif bc == 3:
        nodes_right = domain.get_nodenumber(nx, np.arange(ny + 1))
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    elif bc == 4:
        nodes_right = domain.get_nodenumber(nx, np.arange(ny + 1))
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
        dofs_right = dofs_right[1::2]
    else:
        nodes_right = domain.get_nodenumber(nx, np.arange(1))
        dofs_right = np.repeat(nodes_right * 2, 2, axis=-1) + np.tile(np.arange(2), 1)

    # Node and dof groups
    nodes_left = domain.get_nodenumber(0, np.arange(ny + 1))

    dofs_left = np.repeat(nodes_left * 2, 2, axis=-1) + np.tile(np.arange(2), ny + 1)
    dofs_left_x = dofs_left[0::2]

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_left_x, dofs_right]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    # Setup rhs for loadcase
    f = np.zeros(domain.nnodes * 2)  # Generate a force vector
    f[2 * domain.get_nodenumber(1, ny) + 1] = load

    # Initial design
    s_variables = pym.Signal('x', state=initial_volfrac * np.ones(domain.nel))

    # Setup optimization problem
    fn = pym.Network()

    # Filtering
    s_filtered_variables = fn.append(pym.DensityFilter(s_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    """
    Note the use of SIMPLIN: y = xmin + (1-xmin) * (alpha * x + (alpha - 1) * x^p)
    
    References:
    
    Zhu, J., Zhang, W., & Beckers, P. (2009). Integrated layout design of multi-component system. 
    International Journal for Numerical Methods in Engineering, 78(6), 631–651. https://doi.org/10.1002/nme.2499
    """

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

    if not use_volume_constraint:
        volfrac = 1.0

    # Volume constraint
    s_volume_constraint = fn.append(
        pym.Scaling(s_volume, scaling=scaling_volume_constraint, maxval=volfrac * domain.nel))
    s_volume_constraint.tag = "Volume constraint"

    # Plotting
    module_plotdomain = pym.PlotDomain(s_filtered_variables, domain=domain, saveto="out/design")
    responses = [s_objective, s_volume_constraint]
    plot_signals = responses if use_volume_constraint else [s_objective]
    module_plotiter = pym.PlotIter(plot_signals)
    fn.append(module_plotdomain, module_plotiter)

    # Optimization
    pym.minimize_mma(fn, [s_variables], responses, verbosity=2, maxit=100, move=0.1)
