""" Minimal example for a structural compliance topology optimization """
# flake8: noqa
import pymoto as pym
import numpy as np
from utils.scaling import Scaling

nx, ny = 40,40
xmin, filter_radius, volfrac = 1e-9, 2, 0.3
nu, E = 0.3, 1.0
volume_constraint_scaling = 10
compliance_constraint_value = 0.001
compliance_constraint_scaling = 10
objective_scaling = -10
ka, kb, kf = 4 * (1 - nu), 2 * (1 - 2 * nu), 6 * nu - 3 / 2
el = E / (12 * (1 + nu) * (1 - 2 * nu)) * np.array([
    [ka + kb, -3 / 2, ka / 2 - kb, kf, -ka + kb / 2, -kf, -ka / 2 - kb / 2, 3 / 2],
    [-3 / 2, ka + kb, -kf, -ka + kb / 2, kf, ka / 2 - kb, 3 / 2, -ka / 2 - kb / 2],
    [ka / 2 - kb, -kf, ka + kb, 3 / 2, -ka / 2 - kb / 2, -3 / 2, -ka + kb / 2, kf],
    [kf, -ka + kb / 2, 3 / 2, ka + kb, -3 / 2, -ka / 2 - kb / 2, -kf, ka / 2 - kb],
    [-ka + kb / 2, kf, -ka / 2 - kb / 2, -3 / 2, ka + kb, 3 / 2, ka / 2 - kb, -kf],
    [-kf, ka / 2 - kb, -3 / 2, -ka / 2 - kb / 2, 3 / 2, ka + kb, kf, -ka + kb / 2],
    [-ka / 2 - kb / 2, 3 / 2, -ka + kb / 2, -kf, ka / 2 - kb, kf, ka + kb, -3 / 2],
    [3 / 2, -ka / 2 - kb / 2, kf, ka / 2 - kb, -kf, -ka + kb / 2, -3 / 2, ka + kb]])

# region domain
domain = pym.DomainDefinition(nx, ny)
nodes_left = domain.get_nodenumber(0, np.arange(ny+1))
nodes_right = domain.get_nodenumber(nx, np.arange(ny+1))
dofs_left = np.repeat(nodes_left*2, 2, axis=-1) + np.tile(np.arange(2), ny+1)
dofs_right = np.repeat(nodes_right*2, 2, axis=-1) + np.tile(np.arange(2), ny+1)
dofs_left_horizontal = dofs_left[0::2]
dofs_left_vertical = dofs_left[1::2]
dof_input = dofs_left_vertical[0]
dof_output = dofs_left_vertical[-1]
boundary_dofs = np.union1d(dofs_left_horizontal, dofs_right)
boundary_dofs = np.union1d(boundary_dofs, dof_input)
boundary_dofs = np.union1d(boundary_dofs, dof_output)
# endregion

all_dofs = np.arange(0, 2*domain.nnodes)
prescribed_dofs = boundary_dofs
free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

ndf = len(free_dofs)
ndp = len(prescribed_dofs)

ff = np.zeros((ndf, 2), dtype=float)
u = np.zeros((2*domain.nnodes, 2), dtype=float)
u[dof_input, :] = 1.0
u[dof_output, 0] = -1.0
u[dof_output, 1] = 1.0

up = u[prescribed_dofs, :]

network = pym.Network()

# initial design
signal_variables = pym.Signal('x', state=volfrac * np.ones(domain.nel))

# filtering
signal_filtered_variables = network.append(pym.DensityFilter(signal_variables, domain=domain, radius=filter_radius))

# penalization
signal_penalized_variables = network.append(pym.MathGeneral(signal_filtered_variables, expression=f"{xmin} + {1-xmin}*inp0^3"))

# assembly
signal_stiffness = network.append(pym.AssembleGeneral(signal_penalized_variables, domain=domain, element_matrix=el))

# solve
up = pym.Signal('up', state=up)
ff = pym.Signal('ff', state=ff)
signal_state = network.append(pym.SystemOfEquations([signal_stiffness, ff, up], free=free_dofs, prescribed=prescribed_dofs))

# output displacement
signal_output_displacement = network.append(pym.EinSum([signal_state[0][:, 0], signal_state[1][:, 0]], expression='i,i->'))

# objective
signal_objective = network.append(Scaling([signal_output_displacement], value=objective_scaling))
signal_objective.tag = "OBJ"

# compliances
signal_compliance = network.append(pym.EinSum([signal_state[0][:, 1], signal_state[1][:, 1]], expression='i,i->'))

# compliance constraint input and output
compliance_constraint_string = '{}*(inp0/{} - 1)'.format(compliance_constraint_scaling, compliance_constraint_value, volfrac)
signal_compliance_constraint = network.append(pym.MathGeneral(signal_compliance, expression=compliance_constraint_string))
signal_compliance_constraint.tag = "CC"

# volume
signal_volume = network.append(pym.EinSum(signal_filtered_variables, expression='i->'))
signal_volume.tag = "volume"

# volume constraint
volume_constraint_string = '{}*(inp0/{} - {})'.format(volume_constraint_scaling, domain.nel, volfrac)
signal_volume_constraint = network.append(pym.MathGeneral(signal_volume, expression=volume_constraint_string))
signal_volume_constraint.tag = "VC"

# plotting
module_plotdomain = pym.PlotDomain(signal_filtered_variables, domain=domain, saveto="out/design")
responses = [signal_objective, signal_compliance_constraint, signal_volume_constraint]
module_plotiter = pym.PlotIter(responses)
network.append(module_plotdomain, module_plotiter)

# optimization
pym.minimize_mma(network, [signal_variables], responses, verbosity=2)
