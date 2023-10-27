""" Minimal example for an eigenfrequency topology optimization """
import numpy as np

import pymoto as pym
from modules import VecSet

nx, ny, nz = 50, 30, 0  # Set nz to zero for the 2D problem
xmin = 1e-9
filter_radius = 2.0
volfrac = 0.5
thermal = False  # Thermal only for 2D, not 3D yet. If this is False, static mechanical analysis will be done


class MassInterpolation(pym.Module):
    """ Two-range material interpolation
    For x >= threshold:
        y = rho * x^p1
    For x < threshold:
        y = rho * x^p0 / (t^(p0-p1))
    """

    def _prepare(self, rhoval=1.0, threshold=0.1, p0=6.0, p1=1.0):
        self.rhoval = rhoval
        self.threshold = threshold
        self.p0, self.p1 = p0, p1

    def _response(self, x):
        xx = x ** self.p1
        xx[x < self.threshold] = x[x < self.threshold] ** self.p0 / (self.threshold ** (self.p0 - self.p1))
        return self.rhoval * xx

    def _sensitivity(self, drho):
        x = self.sig_in[0].state
        dx = self.p1 * x ** (self.p1 - 1)
        dx[x < self.threshold] = self.p0 * x[x < self.threshold] ** (self.p0 - 1) / (
                self.threshold ** (self.p0 - self.p1))
        return self.rhoval * dx * drho


if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D structural eigenfrequency analysis

        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        # Calculate boundary dof indices
        boundary_nodes = domain.get_nodenumber(0, np.arange(ny + 1))
        boundary_dofs = np.repeat(boundary_nodes * 2, 2, axis=-1) + np.tile(np.arange(2), len(boundary_nodes))

        # Generate a non-design area that has mass
        nondesign_area = domain.get_elemnumber(*np.meshgrid(range((3 * nx) // 4, nx), range(ny // 4, (ny * 3) // 4)))
        ndof = 2

    else:  # 3D
        domain = pym.DomainDefinition(nx, ny, nz)

        boundary_nodes = domain.get_nodenumber(*np.meshgrid(0, range(ny + 1), range(nz + 1))).flatten()
        boundary_dofs = np.repeat(boundary_nodes * 3, 3, axis=-1) + np.tile(np.arange(3), len(boundary_nodes))

        nondesign_area = domain.get_elemnumber(*np.meshgrid(range((3 * nx) // 4, nx), range(ny // 4, (ny * 3) // 4),
                                                            range(nz // 4, (nz * 3) // 4))).flatten()
        ndof = 3

    if domain.nnodes > 1e+6:
        print("Too many nodes :(")  # Safety, to prevent overloading the memory in your machine
        exit()

    # Make force and design vector, and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    func = pym.Network(print_timing=False)

    # Filter
    sxfilt = func.append(pym.DensityFilter(sx, domain=domain, radius=filter_radius))

    # Set the non-design domain
    sxndd = func.append(VecSet(sxfilt, indices=nondesign_area, value=1.0))

    sx_analysis = sxndd

    # Show the design on the screen as it optimizes
    func.append(pym.PlotDomain(sx_analysis, domain=domain, saveto="out/design", clim=[0, 1]))

    # SIMP material interpolation
    sSIMP = func.append(pym.MathGeneral(sx_analysis, expression=f"{xmin} + {1.0 - xmin}*inp0^3"))
    sDENS = func.append(MassInterpolation(sx_analysis))

    # System matrix assembly module
    sK = func.append(pym.AssembleStiffness(sSIMP, domain=domain, bc=boundary_dofs))
    sM = func.append(pym.AssembleMass(sDENS, domain=domain, bc=boundary_dofs))

    # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
    slams, seigvec = func.append(pym.EigenSolve([sK, sM], hermitian=True, nmodes=3))

    # Output the design, deformation, and force field to a Paraview file
    func.append(pym.WriteToVTI([sx_analysis], domain=domain, saveto='out/dat.vti'))

    # Get harmonic mean of three lowest eigenvalues
    sharm = func.append(pym.MathGeneral([slams[0], slams[1], slams[2]], expression='1/inp0 + 1/inp1 + 1/inp2'))

    # MMA needs correct scaling of the objective
    sg0 = func.append(pym.Scaling(sharm, scaling=100.0))
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

    # Do the optimization with MMA
    pym.minimize_mma(func, [sx], [sg0, sg1], verbosity=2)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final eigenfrequencies obtained are {np.sqrt(slams.state)}")
