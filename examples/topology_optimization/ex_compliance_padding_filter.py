""" Convolution filter with domain padding
==========================================

Not only is the convolution filter, but allows various types of padding for control of the boundary of the design

The :py:class:`pymoto.FilterConv` module implements a padded-domain filter, which is demonstrated currently for a 
compliance topology optimization with a padded-domain filter, which avoids the design 'sticking' to the boundary. By 
turning the padding on or off in this example, the effect can clearly be seen.

This example is based on the robust formulation with Heaviside projections as in 
:ref:`sphx_glr_auto_examples_topology_optimization_ex_compliance_robust.py`.
"""
import numpy as np
import pymoto as pym

nx, ny, nz = 120, 40, 0  # Set nz to zero for the 2D problem
xmin = 1e-9
filter_radius = 5.0
volfrac = 0.5


class Continuation(pym.Module):
    """ Module that generates a continuated value """
    def __init__(self, start=0.0, stop=1.0, nsteps=80, stepstart=10):
        self.startval = start
        self.endval = stop
        self.dval = (stop - start) / nsteps
        self.nstart = stepstart
        self.iter = -1
        self.val = self.startval

    def __call__(self):
        if (self.val < self.endval) and (self.iter > self.nstart):
            self.val += self.dval

        self.val = np.clip(self.val, min(self.startval, self.endval), max(self.startval, self.endval))
        self.iter += 1
        return self.val


if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D analysis
        ndof = 2

        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        # Get node numbers for the boundary condition
        boundary_nodes = domain.nodes[0, ny//3:-ny//3].flatten()

        # Calculate boundary dof indices
        boundary_dofs = np.repeat(boundary_nodes * ndof, ndof, axis=-1) + np.tile(np.arange(ndof), len(boundary_nodes))

        # Which dofs to put a force on? The 1 is added for a force in y-direction (x-direction would be zero)
        force_dofs = ndof * domain.nodes[nx, ny//2] + 1
    else:
        ndof = 3
        domain = pym.DomainDefinition(nx, ny, nz)
        boundary_nodes = domain.nodes[0, ny//3:-ny//3, nz//3:-nz//3].flatten()
        boundary_dofs = np.repeat(boundary_nodes * ndof, ndof, axis=-1) + np.tile(np.arange(ndof), len(boundary_nodes))
        force_dofs = ndof * domain.nodes[nx, ny // 2, nz // 2] + 2  # Z-direction

    if domain.nnodes > 1e+6:
        print("Too many nodes :(")  # Safety, to prevent accidentally overloading the memory in your machine
        exit()

    # Generate a force vector
    f = np.zeros(domain.nnodes * ndof)
    f[force_dofs] = 1.0  # Uniform force of 1.0 at all selected dofs

    # Make design vector and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    with pym.Network() as func:
        do_padding = True  # Switch padding on or off
        if do_padding:
            ''' Filter with padding to avoid the design 'sticking' to the boundaries
            Set densities just outside the boundaries to 0, except for the xmax. The xmax boundary is kept symmetric, 
            which is the standard behavior of a density filter. This is done to allow full solid to be formed here in 
            case the mechanical load is at the boundary.
            '''
            m_filt = pym.FilterConv(domain, radius=filter_radius,
                                    xmin_bc=0, xmax_bc='symmetric',
                                    ymin_bc=0, ymax_bc=0,
                                    zmin_bc=0, zmax_bc=0)
            sxfilt = m_filt(sx)
             
            ''' Furthermore, a density of 1 is set just outside the boundary condition at xmin for correct filtering 
            This is done manually by overriding the padded values of the filter.
            '''
            xrange = np.arange(m_filt.pad_sizes[0])
            yrange = m_filt.pad_sizes[1] + np.arange(domain.nely//3, domain.nely - domain.nely//3)
            if domain.dim < 3:
                zrange = np.array([0])
            else:
                zrange = m_filt.pad_sizes[2] + np.arange(domain.nelz//3, domain.nelz - domain.nelz//3)
            ex, ey, ez = np.meshgrid(xrange, yrange, zrange)
            m_filt.override_padded_values((ex, ey, ez), 1.0)
            ''' Everything from here is the same as in ex_compliance_robust.py '''
        else:
            # Use convolution filter as regular density filter (without padding) to see the difference
            sxfilt = pym.FilterConv(domain, radius=filter_radius)(sx)

        # Heaviside projections
        etaDi, etaNo, etaEr = 0.3, 0.5, 0.7
        sBeta = Continuation(start=1., stop=20.0, stepstart=10)()
        sBeta.tag = "beta"

        pym.Print()(sBeta)  # Print the beta value each iteration

        heaviside = "(tanh(inp1 * {0}) + tanh(inp1 * (inp0 - {0}))) / (tanh(inp1 * {0}) + tanh(inp1 * (1 - {0})))"
        sxNom = pym.MathGeneral(heaviside.format(etaNo))(sxfilt, sBeta)
        sxEr = pym.MathGeneral(heaviside.format(etaEr))(sxfilt, sBeta)
        sxDi = pym.MathGeneral(heaviside.format(etaDi))(sxfilt, sBeta)
        sxNom.tag = "xnominal"
        sxEr.tag = "xeroded"
        sxDi.tag = "xdilated"

        sx_analysis = sxEr  # Use this design for analysis

        # Show the (nominal) design on the screen as it optimizes
        pym.PlotDomain(domain, saveto="out/design", clim=[0, 1])(sxNom)

        # SIMP material interpolation
        sSIMP = pym.MathGeneral(f"{xmin} + {1.0 - xmin}*inp0^3")(sx_analysis)

        # System matrix assembly module
        sK = pym.AssembleStiffness(domain, bc=boundary_dofs)(sSIMP)

        # Linear system solver. The linear solver can be chosen by uncommenting any of the following lines.
        solver = None  # Default (solver is automatically chosen based on matrix properties)
        # solver = pym.solvers.SolverSparsePardiso()  # Requires Intel MKL installed
        # solver = pym.solvers.SolverSparseCholeskyCVXOPT()  # Requires cvxopt installed
        # solver = pym.solvers.SolverSparseCholeskyScikit()  # Requires scikit installed
        su = pym.LinSolve(hermitian=True, solver=solver)(sK, f)

        # Compliance calculation c = f^T u
        scompl = pym.EinSum('i,i->')(su, f)
        scompl.tag = 'compliance'

        # MMA needs correct scaling of the objective
        sg0 = pym.Scaling(scaling=100.0)(scompl)
        sg0.tag = "objective"

        # Calculate the volume of the domain by adding all design densities together
        svol = pym.EinSum('i->')(sxNom)
        svol.tag = 'volume'

        # Volume constraint
        sg1 = pym.MathGeneral(f'10*(inp0/{domain.nel} - {volfrac})')(svol)
        sg1.tag = "volume constraint"

    # Maybe you want to check the design-sensitivities?
    do_finite_difference = False
    if do_finite_difference:
        pym.finite_difference(sx, [sg0, sg1], dx=1e-4)
        exit()

    with func:
        pym.PlotIter()(sg0, sg1)  # Plot iteration history

    # Do the optimization with MMA
    pym.minimize_mma(sx, [sg0, sg1], function=func)

    # Do the optimization with OC
    # pym.minimize_oc(sx, sg0, function=func)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum displacement is {max(np.absolute(su.state))}")
