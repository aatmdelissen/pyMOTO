""" Example for a compliance topology optimization, including robust formulation """
import pymoto as pym
import numpy as np

nx, ny = 100, 40
xmin = 1e-6
filter_radius = 3.0
volfrac = 0.5

if __name__ == "__main__":
    print(__doc__)

    # --- SETUP ---
    # Generate a grid
    domain = pym.DomainDefinition(nx, ny)

    # Chose which physics to solve: structural or thermal
    physics = "structural"  # "thermal"
    if physics == "structural":
        # STRUCTURAL
        # Calculate boundary dof indices
        boundary_nodes = domain.get_nodenumber(0, np.arange(ny+1))
        boundary_dofs = np.repeat(boundary_nodes * 2, 2, axis=-1) + np.tile(np.arange(2), len(boundary_nodes))

        # Generate a force vector
        force_dofs = 2*domain.get_nodenumber(nx, ny//2)
        ndof = 2  # Number of displacements per node

        # Set padded area for the filter
        pad = domain.get_elemnumber(*np.meshgrid(np.arange(filter_radius), np.arange(ny)))

        # Compute element stiffness matrix
        nu, E = 0.3, 1.0
        lx, ly, lz = 1.0, 1.0, 1.0
        c = ly / lx
        ka = (4 * c) * (1 - nu)
        kc = (4 / c) * (1 - nu)
        kd = (2 * c) * (1 - 2 * nu)
        kb = (2 / c) * (1 - 2 * nu)
        kf = 6 * nu - 3 / 2
        ke = E * lz / (12 * (1 + nu) * (1 - 2 * nu))

        el = ke * np.array([
            [ka + kb, -3 / 2, ka / 2 - kb, kf, -ka + kb / 2, -kf, -ka / 2 - kb / 2, 3 / 2],
            [-3 / 2, kc + kd, -kf, -kc + kd / 2, kf, kc / 2 - kd, 3 / 2, -kc / 2 - kd / 2],
            [ka / 2 - kb, -kf, ka + kb, 3 / 2, -ka / 2 - kb / 2, -3 / 2, -ka + kb / 2, kf],
            [kf, -kc + kd / 2, 3 / 2, kc + kd, -3 / 2, -kc / 2 - kd / 2, -kf, kc / 2 - kd],
            [-ka + kb / 2, kf, -ka / 2 - kb / 2, -3 / 2, ka + kb, 3 / 2, ka / 2 - kb, -kf],
            [-kf, kc / 2 - kd, -3 / 2, -kc / 2 - kd / 2, 3 / 2, kc + kd, kf, -kc + kd / 2],
            [-ka / 2 - kb / 2, 3 / 2, -ka + kb / 2, -kf, ka / 2 - kb, kf, ka + kb, -3 / 2],
            [3 / 2, -kc / 2 - kd / 2, kf, kc / 2 - kd, -kf, -kc + kd / 2, -3 / 2, kc + kd],
        ])

    elif physics == "thermal":
        # THERMAL
        # Get dof numbers at the boundary
        boundary_dofs = domain.get_nodenumber(0, np.arange(ny//4, (3*ny)//4))

        # Make a force vector
        force_dofs = domain.get_nodenumber(*np.meshgrid(np.arange(nx // 4, nx + 1), np.arange(ny + 1)))
        ndof = 1  # Number of dofs per node

        # Set padded area for the filter
        pad = domain.get_elemnumber(*np.meshgrid(np.arange(filter_radius), np.arange(ny)))

        # Element conductivity matrix
        el = 1 / 6 * np.array([
            [8, -2, -2, -4],
            [-2, 8, -4, -2],
            [-2, -4, 8, -2],
            [-4, -2, -2, 8]
        ])
    else:
        raise RuntimeError("Unknown physics: {}".format(physics))

    # Make force and design vector, and fill with initial values
    f = np.zeros(domain.nnodes*ndof)
    f[force_dofs] = 1.0

    sf = pym.Signal('f', state=f)
    sx = pym.Signal('x', state=np.ones(domain.nel)*volfrac)

    # Start building the modular network
    fn = pym.Network()

    # Filter
    sxfilt = pym.Signal('xfiltered')
    fn.append(pym.Density(sx, sxfilt, domain=domain, radius=filter_radius, nonpadding=pad))

    # Module that generates a continuated value
    class Continuation(pym.Module):
        def _prepare(self, start=0.0, stop=1.0, nsteps=80, stepstart=10):
            self.startval = start
            self.endval = stop
            self.dval = (stop - start)/nsteps
            self.nstart = stepstart
            self.iter = -1
            self.val = self.startval

        def _response(self):
            if (self.val < self.endval and self.iter > self.nstart):
                self.val += self.dval

            self.val = np.clip(self.val, min(self.startval, self.endval), max(self.startval, self.endval))
            print(self.sig_out[0].tag, ' = ', self.val)
            self.iter += 1
            return self.val

        def _sensitivity(self, *args):
            pass

    etaDi, etaNo, etaEr = 0.3, 0.5, 0.7
    sBeta = pym.Signal("beta")
    fn.append(Continuation([], sBeta, start=1.0, stop=20.0, stepstart=10))

    sxNom = pym.Signal("xnominal")
    sxEr = pym.Signal("xeroded")
    sxDi = pym.Signal("xdilated")

    heaviside = "(tanh(inp1 * {0}) + tanh(inp1 * (inp0 - {0}))) / (tanh(inp1 * {0}) + tanh(inp1 * (1 - {0})))"
    fn.append(pym.MathGeneral([sxfilt, sBeta], sxNom, expression=heaviside.format(etaNo)))
    fn.append(pym.MathGeneral([sxfilt, sBeta], sxEr, expression=heaviside.format(etaEr)))
    fn.append(pym.MathGeneral([sxfilt, sBeta], sxDi, expression=heaviside.format(etaDi)))

    # SIMP material interpolation
    sSIMP = pym.Signal('x3')
    fn.append(pym.MathGeneral(sxEr, sSIMP, expression="{0} + (1.0-{0})*inp0^3".format(xmin)))

    # Add stiffness assembly module
    sK = pym.Signal('K')
    fn.append(pym.AssembleGeneral(sSIMP, sK, domain=domain, element_matrix=el, bc=boundary_dofs))

    # Linear system solver
    su = pym.Signal('u')
    fn.append(pym.LinSolve([sK, sf], su))

    # Compliance calculation
    sc = pym.Signal('compliance')
    fn.append(pym.EinSum([su, sf], sc, expression='i,i->'))

    # Define a new module for scaling
    class ScaleTo(pym.Module):
        def _prepare(self, val=100.0):
            self.targetval = val
            self.initval = None

        def _response(self, x):
            if self.initval is None:
                self.initval = x
            return x/self.initval * self.targetval

        def _sensitivity(self, dfdy):
            return dfdy/self.initval * self.targetval

    # Objective scaling
    sg0 = pym.Signal('objective')
    fn.append(ScaleTo(sc, sg0, val=100.0))

    # Plot design
    fn.append(pym.PlotDomain(sxNom, domain=domain, saveto="out/design"))

    # Volume
    svol = pym.Signal('vol')
    fn.append(pym.EinSum(sxfilt, svol, expression='i->'))

    # Volume constraint
    sg1 = pym.Signal('volconstraint')
    fn.append(pym.MathGeneral(svol, sg1, expression='10*(inp0/{} - {})'.format(domain.nel, volfrac)))

    fn.append(pym.PlotIter([sg0]))

    # --- OPTIMIZATION ---
    # pym.finite_difference(func, sx, sg0)  # Note that the FD won't be correct due to the continuation changing values
    pym.minimize_oc(fn, [sx], sg0, tolx=0.0, tolf=0.0)
    # pym.minimize_mma(fn, [sx], [sg0, sg1], tolx=0.0, tolf=0.0)  # TODO NLopt MMA version also doesnt work here

