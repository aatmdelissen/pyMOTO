""" Compliance topology optimization with a padded-domain filter, which avoids the design 'sticking' to the boundary """
import numpy as np

import pymoto as pym

nx, ny, nz = 120, 40, 0  # Set nz to zero for the 2D problem
xmin = 1e-9
filter_radius = 5.0
volfrac = 0.5


class Continuation(pym.Module):
    """ Module that generates a continuated value """
    def _prepare(self, start=0.0, stop=1.0, nsteps=80, stepstart=10):
        self.startval = start
        self.endval = stop
        self.dval = (stop - start) / nsteps
        self.nstart = stepstart
        self.iter = -1
        self.val = self.startval

    def _response(self):
        if (self.val < self.endval) and (self.iter > self.nstart):
            self.val += self.dval

        self.val = np.clip(self.val, min(self.startval, self.endval), max(self.startval, self.endval))
        print(self.sig_out[0].tag, ' = ', self.val)
        self.iter += 1
        return self.val

    def _sensitivity(self, *args):
        pass


if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D analysis
        ndof = 2

        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        # Get node numbers for the boundary condition
        boundary_nodes = domain.get_nodenumber(0, np.arange(ny // 3, (ny+1) - ny//3))

        # Calculate boundary dof indices
        boundary_dofs = np.repeat(boundary_nodes * ndof, ndof, axis=-1) + np.tile(np.arange(ndof), len(boundary_nodes))

        # Which dofs to put a force on? The 1 is added for a force in y-direction (x-direction would be zero)
        force_dofs = ndof * domain.get_nodenumber(nx, ny // 2) + 1

    else:
        ndof = 3
        domain = pym.DomainDefinition(nx, ny, nz)
        boundary_nodes = domain.get_nodenumber(*np.meshgrid(0, np.arange(ny // 3, (ny+1) - ny//3), np.arange(nz // 3, (nz+1) - nz//3))).flatten()
        boundary_dofs = np.repeat(boundary_nodes * ndof, ndof, axis=-1) + np.tile(np.arange(ndof), len(boundary_nodes))
        force_dofs = ndof * domain.get_nodenumber(nx, ny // 2, nz // 2) + 2  # Z-direction

    if domain.nnodes > 1e+6:
        print("Too many nodes :(")  # Safety, to prevent overloading the memory in your machine
        exit()

    # Generate a force vector
    f = np.zeros(domain.nnodes * ndof)
    f[force_dofs] = 1.0  # Uniform force of 1.0 at all selected dofs

    # Make force and design vector, and fill with initial values
    sf = pym.Signal('f', state=f)
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    func = pym.Network(print_timing=False)

    do_padding = True  # Switch padding on or off
    if do_padding:
        ''' Filter with padding to avoid the design 'sticking' to the boundaries
        Set densities just outside the boundaries to 0, except for the xmax. The xmax boundary is kept symmetric, which 
        is the standard behavior of a density filter. This is done to allow full solid to be formed here in case the 
        mechanical load is at the boundary.
        '''
        sxfilt = func.append(pym.FilterConv(sx, domain=domain, radius=filter_radius,
                                            xmin_bc=0, xmax_bc='symmetric',
                                            ymin_bc=0, ymax_bc=0,
                                            zmin_bc=0, zmax_bc=0))
        m_filt = func[-1]
        ''' Furthermore, a density of 1 is set just outside the boundary condition at xmin for correct filtering '''
        xrange = np.arange(m_filt.pad_sizes[0])
        yrange = m_filt.pad_sizes[1] + np.arange(domain.nely//3, domain.nely - domain.nely//3)
        zrange = np.array([0]) if domain.dim < 3 else m_filt.pad_sizes[2] + np.arange(domain.nelz//3, domain.nelz - domain.nelz//3)
        ex, ey, ez = np.meshgrid(xrange, yrange, zrange)
        m_filt.override_padded_values((ex, ey, ez), 1.0)
        ''' Everything from here is the same as in ex_compliance.py '''
    else:
        # Use regular density filter (without padding) to see the difference
        sxfilt = func.append(pym.DensityFilter(sx, domain=domain, radius=filter_radius))

    # Heaviside projections
    etaDi, etaNo, etaEr = 0.3, 0.5, 0.7
    sBeta = pym.Signal("beta")
    func.append(Continuation([], sBeta, start=1.0, stop=20.0, stepstart=10))

    sxNom = pym.Signal("xnominal")
    sxEr = pym.Signal("xeroded")
    sxDi = pym.Signal("xdilated")

    heaviside = "(tanh(inp1 * {0}) + tanh(inp1 * (inp0 - {0}))) / (tanh(inp1 * {0}) + tanh(inp1 * (1 - {0})))"
    func.append(pym.MathGeneral([sxfilt, sBeta], sxNom, expression=heaviside.format(etaNo)))
    func.append(pym.MathGeneral([sxfilt, sBeta], sxEr, expression=heaviside.format(etaEr)))
    func.append(pym.MathGeneral([sxfilt, sBeta], sxDi, expression=heaviside.format(etaDi)))

    sx_analysis = sxEr

    # Show the design on the screen as it optimizes
    func.append(pym.PlotDomain(sx_analysis, domain=domain, saveto="out/design", clim=[0, 1]))

    # SIMP material interpolation
    sSIMP = func.append(pym.MathGeneral(sx_analysis, expression=f"{xmin} + {1.0 - xmin}*inp0^3"))

    # System matrix assembly module
    sK = func.append(pym.AssembleStiffness(sSIMP, domain=domain, bc=boundary_dofs))

    # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
    solver = None  # Default (solver is automatically chosen based on matrix properties)
    # solver = pym.solvers.SolverSparsePardiso()  # Requires Intel MKL installed
    # solver = pym.solvers.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
    # solver = pym.solvers.SolverSparseCholeskyScikit()  # Requires scikit installed
    su = func.append(pym.LinSolve([sK, sf], hermitian=True, solver=solver))

    # Compliance calculation c = f^T u
    scompl = func.append(pym.EinSum([su, sf], expression='i,i->'))
    scompl.tag = 'compliance'

    # MMA needs correct scaling of the objective
    sg0 = func.append(pym.Scaling(scompl, scaling=100.0))
    sg0.tag = "objective"

    # Calculate the volume of the domain by adding all design densities together
    svol = func.append(pym.EinSum(sxNom, expression='i->'))
    svol.tag = 'volume'

    # Volume constraint
    sg1 = func.append(pym.MathGeneral(svol, expression=f'10*(inp0/{domain.nel} - {volfrac})'))
    sg1.tag = "volume constraint"

    # Maybe you want to check the design-sensitivities?
    do_finite_difference = False
    if do_finite_difference:
        pym.finite_difference(func, sx, [sg0, sg1], dx=1e-4)
        exit()

    func.append(pym.PlotIter([sg0, sg1]))  # Plot iteration history

    # Do the optimization with MMA
    pym.minimize_mma(func, [sx], [sg0, sg1])

    # Do the optimization with OC
    # pym.minimize_oc(func, sx, sg0)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum displacement is {max(np.absolute(su.state))}")
