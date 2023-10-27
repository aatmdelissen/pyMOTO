"""
Example of the design of a flexure using topology optimization with:
(i) maximum shear stiffness
(ii) constrainted maximum stiffness in axial stiffness
(iii) (optional) constrained maximum use of material (volume constraint)

Implemented by @artofscience (s.koppen@tudelft.nl) based on:

Koppen, S., Langelaar, M., & van Keulen, F. (2022).
A simple and versatile topology optimization formulation for flexure synthesis.
Mechanism and Machine Theory, 172, 104743.
DOI: http://dx.doi.org/10.1016/j.mechmachtheory.2022.104743
"""
import numpy as np

# flake8: noqa
import pymoto as pym
from modules import Symmetry, Stress, VonMises, ConstraintAggregation


def flexure(nx: int = 20, ny: int = 20, doc: str = 'tx', dof: str = 'ty', emax: float = 1.0,
            filter_radius: float = 2.0, E: float = 100.0, nu: float = 0.3, xmin: float = 1e-9,
            volume_constraint = 0.3, initial_volfrac: float = 0.2, use_symmetry=False, stress_constraint=None):

    scaling_objective = 10.0
    scaling_compliance_constraint = 10.0
    scaling_volume_constraint = 10.0

    # region preproc
    degrees = ('tx', 'ty', 'rz')
    deg = [degrees.index(doc), degrees.index(dof)]

    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_top = domain.get_nodenumber(np.arange(nx + 1), ny)
    nodes_bottom = domain.get_nodenumber(np.arange(nx + 1), 0)

    dofs_top = np.repeat(nodes_top * 2, 2, axis=-1) + np.tile(np.arange(2), nx + 1)
    dofs_bottom = np.repeat(nodes_bottom * 2, 2, axis=-1) + np.tile(np.arange(2), nx + 1)

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_bottom, dofs_top]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    dofs_top_x = dofs_top[0::2]
    dofs_top_y = dofs_top[1::2]
    # endregion

    v = np.zeros((2 * domain.nnodes, 3), dtype=float)
    v[dofs_top_x, 0] = 1.0  # tx and rz
    v[dofs_top_x, 2] = 1.0 * ny / nx
    v[dofs_top_y, 1] = 1.0  # ty
    v[dofs_top_y, 2] = np.linspace(1, -1, nx + 1)  # rz
    u = v[:, deg]

    # Initial design
    signal_variables = pym.Signal('x', state=initial_volfrac * np.ones(domain.nel))

    # Setup optimization problem
    network = pym.Network()

    # Force symmetry
    if use_symmetry:
        signal_variables_symmetric = network.append(Symmetry(signal_variables, domain=domain))
        signal_filtered_variables = network.append(
            pym.DensityFilter(signal_variables_symmetric, domain=domain, radius=filter_radius))
    else:
        signal_filtered_variables = network.append(
            pym.DensityFilter(signal_variables, domain=domain, radius=filter_radius))

    # SIMP penalization
    signal_penalized_variables = network.append(
        pym.MathGeneral(signal_filtered_variables, expression=f"{xmin} + {1 - xmin}*(0.01*inp0 + 0.99*inp0^3)"))

    # Assembly
    signal_stiffness = network.append(
        pym.AssembleStiffness(signal_penalized_variables, domain=domain, e_modulus=E, poisson_ratio=nu))

    # Solve system of equations for the two loadcases
    up = pym.Signal('up', state=u[prescribed_dofs, :])
    ff = pym.Signal('ff', state=np.zeros((free_dofs.size, 2), dtype=float))
    signal_state = network.append(
        pym.SystemOfEquations([signal_stiffness, ff, up], free=free_dofs, prescribed=prescribed_dofs))

    # Compliance
    signal_compliances = network.append(pym.EinSum([signal_state[0], signal_state[1]], expression='ij,ij->j'))

    # Objective function
    signal_objective = network.append(pym.Scaling([signal_compliances[0]], scaling=-1.0 * scaling_objective))
    signal_objective.tag = "Objective"

    # Compliance constraint
    signal_compliance_constraint = network.append(
        pym.Scaling(signal_compliances[1], scaling=scaling_compliance_constraint, maxval=emax))
    signal_compliance_constraint.tag = "Compliance constraint"

    # Plotting
    network.append(pym.PlotDomain(signal_filtered_variables, domain=domain, saveto="out/design"))
    responses = [signal_objective, signal_compliance_constraint]
    plot_signals = responses.copy()

    # Calculate stress
    s_stress = network.append(Stress([signal_state[0][:, 1]], domain=domain))
    s_stress_vm = network.append(VonMises([s_stress]))

    if stress_constraint:
        s_stress_constraints = network.append(pym.Scaling([s_stress_vm], maxval=stress_constraint, scaling=1.0))

        s_stress_constraints_scaled = network.append(
            pym.EinSum([signal_filtered_variables, s_stress_constraints], expression='i,i->i'))

        s_stress_constraint = network.append(ConstraintAggregation([s_stress_constraints_scaled], P=10))
        s_stress_constraint.tag = "Stress constraint"

        responses.append(s_stress_constraint)
        plot_signals.append(s_stress_constraint)

    # Plotting
    s_stress_scaled = network.append(pym.EinSum([signal_filtered_variables, s_stress_vm], expression='i,i->i'))
    module_plotstress = pym.PlotDomain(s_stress_scaled, domain=domain, cmap='jet')
    network.append(module_plotstress)

    if volume_constraint:
        # Volume
        s_volume = network.append(pym.EinSum(signal_filtered_variables, expression='i->'))

        # Volume constraint
        s_volume_constraint = network.append(
            pym.Scaling(s_volume, scaling=scaling_volume_constraint, maxval=volume_constraint * domain.nel))
        s_volume_constraint.tag = "Volume constraint"
        responses.append(s_volume_constraint)
        plot_signals.append(s_volume_constraint)

    network.append(pym.PlotIter(plot_signals))

    # Optimization
    pym.minimize_mma(network, [signal_variables], responses, verbosity=2, maxit=200)


if __name__ == "__main__":
    # flexure(100, 120, 'rz', 'ty', 0.1)  # axial spring, no rotation
    # flexure(100, 100, 'tx', 'ty', 1, use_symmetry=True, use_stress_constraint=True, max_stress=0.01)  # axial spring, no shear
    # flexure(100, 100, 'ty', 'tx', 0.01)  # parallel guiding system
    # flexure(100, 100, 'rz', 'tx', 0.01)  # parallel guiding system 2
    flexure(100, 100, 'ty', 'rz', 0.01, volume_constraint=0.3, stress_constraint=0.001)  # notch hinge
    # flexure(100, 50, 'tx', 'rz', 0.01, use_symmetry=True)  # notch hinge 2
