""" Minimal example for a compliance topology optimization """
import pymodular as pym
import numpy as np

nx, ny, nz = 60, 30, 0  # Set nz to zero for the 2D problem
xmin = 1e-9
filter_radius = 2.0
volfrac = 0.5
thermal = False  # Thermal only for 2D, not 3D yet. If this is False, static mechanical analysis will be done


class Scaling(pym.Module):
    """
    Quick module that scales to a given value on the first iteration.
    This is useful, for instance, for MMA where the objective must be scaled in a certain way for good convergence
    """
    def _prepare(self, value):
        self.value = value

    def _response(self, x):
        if not hasattr(self, 'sf'):
            self.sf = self.value/x
        return x * self.sf

    def _sensitivity(self, dy):
        return dy * self.sf


if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D analysis
        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        if thermal:
            # Get dof numbers at the boundary
            boundary_dofs = domain.get_nodenumber(0, np.arange(ny // 4, (ny+1) - ny//4))

            # Make a force vector
            force_dofs = domain.get_nodenumber(*np.meshgrid(np.arange(1, nx + 1), np.arange(ny + 1)))

            # Element conductivity matrix
            el = 1 / 6 * np.array([
                [+8., -2., -2., -4.],
                [-2., +8., -4., -2.],
                [-2., -4., +8., -2.],
                [-4., -2., -2., +8.]
            ])
            ndof = 1  # Number of dofs per node

        else:  # Mechanical
            # Calculate boundary dof indices
            boundary_nodes = domain.get_nodenumber(0, np.arange(ny+1))
            boundary_dofs = np.repeat(boundary_nodes * 2, 2, axis=-1) + np.tile(np.arange(2), len(boundary_nodes))

            # Which dofs to put a force on? The 1 is added for a force in y-direction (x-direction would be zero)
            force_dofs = domain.dim*domain.get_nodenumber(nx, ny//2) + 1
            ndof = 2

    else:
        domain = pym.DomainDefinition(nx, ny, ny)

        if thermal:
            raise RuntimeError("Thermal only defined in 2D!")  # TODO
        else:
            boundary_nodes = domain.get_nodenumber(*np.meshgrid(0, range(ny+1), range(ny+1))).flatten()
            boundary_dofs = np.repeat(boundary_nodes * 3, 3, axis=-1) + np.tile(np.arange(3), len(boundary_nodes))

            force_dofs = domain.dim*domain.get_nodenumber(nx, ny//2, ny//2)+2  # Z-direction
            ndof = 3

    if domain.nnodes > 1e+6:
        print("Too many nodes :(")  # Safety, to prevent overloading the memory in your machine
        exit()

    # Generate a force vector
    f = np.zeros(domain.nnodes*ndof)
    f[force_dofs] = 1.0  # Uniform force of 1.0 at all selected dofs

    # Make force and design vector, and fill with initial values
    sf = pym.Signal('f', state=f)
    sx = pym.Signal('x', state=np.ones(domain.nel)*volfrac)

    # Start building the modular network
    func = pym.Network(print_timing=False)

    # Filter
    sxfilt = func.append(pym.Density(sx, domain=domain, radius=filter_radius))
    sx_analysis = sxfilt

    # Show the design on the screen as it optimizes
    func.append(pym.PlotDomain(sx_analysis, domain=domain, saveto="out/design", clim=[0, 1]))

    # SIMP material interpolation
    sSIMP = func.append(pym.MathGeneral(sx_analysis, expression=f"{xmin} + {1.0-xmin}*inp0^3"))

    # System matrix assembly module
    if thermal:
        sK = func.append(pym.AssembleGeneral(sSIMP, domain=domain, element_matrix=el, bc=boundary_dofs))
    else:
        sK = func.append(pym.AssembleStiffness(sSIMP, domain=domain, bc=boundary_dofs))

    # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
    solver = None  # Default (automatic search)
    # solver = pym.SolverSparsePardiso()  # Requires Intel MKL installed
    # solver = pym.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
    # solver = pym.SolverSparseCholeskyScikit()  # Requires scikit installed
    su = func.append(pym.LinSolve([sK, sf], hermitian=True, solver=solver))

    # Output the design, deformation, and force field to a Paraview file
    func.append(pym.WriteToParaview([sx_analysis, su, sf], domain=domain, saveto='out/dat.vti'))

    # Compliance calculation c = f^T u
    scompl = func.append(pym.EinSum([su, sf], expression='i,i->'))
    scompl.tag = 'compliance'

    # MMA needs correct scaling of the objective
    sg0 = func.append(Scaling(scompl, value=100.0))
    sg0.tag = "objective"

    # Calculate the volume of the domain by adding all design densities together
    svol = func.append(pym.EinSum(sx_analysis, expression='i->'))
    svol.tag = 'volume'

    # Volume constraint
    sg1 = func.append(pym.MathGeneral(svol, expression='10*(inp0/{} - {})'.format(domain.nel, volfrac)))
    sg1.tag = "volume constraint"

    # Maybe you want to check the design-sensitivities?
    do_finite_difference = False
    if do_finite_difference:
        pym.finite_difference(func, sx, [sg0, sg1], dx=1e-4)
        exit()

    func.append(pym.PlotIter([sg0, sg1]))  # Plot iteration history

    # Do the optimization with MMA (requires nlopt package)
    # pym.minimize_mma(func, [sx], [sg0, sg1])  # TODO does not work correctly for the thermal case

    # Do the optimization with OC
    pym.minimize_oc(func, [sx], sg0)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum {'temperature' if thermal else 'displacement'} is {max(np.absolute(su.state))}")


