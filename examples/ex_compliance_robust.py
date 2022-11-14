""" Example for a compliance topology optimization """
import pymodular as pym
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    # --- SETUP ---
    # Generate a grid
    nx, ny = 100, 40
    domain = pym.DomainDefinition(nx, ny)

    xmin = 1e-3
    filter_radius = 1.5

    physics = "structural"  # "thermal"
    if physics == "structural":
        # STRUCTURAL
        # Calculate boundary dof indices
        boundary_nodes = domain.get_nodenumber(0, np.arange(ny+1))
        boundary_dofs = np.repeat(boundary_nodes * 2, 2, axis=-1) + np.tile(np.arange(2), len(boundary_nodes))

        # Generate a force vector
        force_dofs = 2*domain.get_nodenumber(nx, ny//2)
        f = np.zeros(domain.nnodes*2)
        f[force_dofs] = 1.0

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
        f = np.zeros(domain.nnodes)
        f[force_dofs] = 1.0

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
    sf = pym.Signal('f', state=f)
    sx = pym.Signal('x', state=np.ones(domain.nel)*0.5)

    # Start building the modular network
    mods = []

    # Filter
    sxfilt = pym.Signal('xfiltered')
    mods.append(pym.Density(sx, sxfilt, domain=domain, radius=filter_radius, nonpadding=pad))

    # Robust
    class Continuation(pym.Module):
        def _prepare(self, start=0.0, stop=1.0, nsteps=80, stepstart=10):
            self.startval = start
            self.endval = stop
            self.dval = (stop - start)/nsteps
            self.nstart = stepstart
            self.iter = -1
            self.val = self.startval

        def _response(self):
            self.iter += 1
            if self.iter % self.nstart == 0:
                self.val *= 2
                self.val = np.clip(self.val, min(self.startval, self.endval), max(self.startval, self.endval))
            print(self.sig_out[0].tag, ' = ', self.val)
            return self.val

        def _sensitivity(self, *args):
            pass

    etaDi, etaNo, etaEr = 0.3, 0.5, 0.7
    beta = 0.001
    sxNom = pym.Signal("xnominal")
    sxEr = pym.Signal("xeroded")
    sxDi = pym.Signal("xdilated")

    heaviside = "(tanh({1} * {0}) + tanh({1} * (inp0 - {0}))) / (tanh({1} * {0}) + tanh({1} * (1 - {0})))"
    mods.append(pym.MathGeneral(sxfilt, sxNom, expression=heaviside.format(etaNo, beta)))
    mods.append(pym.MathGeneral(sxfilt, sxEr, expression=heaviside.format(etaEr, beta)))
    mods.append(pym.MathGeneral(sxfilt, sxDi, expression=heaviside.format(etaDi, beta)))

    # SIMP material interpolation
    sSIMP = pym.Signal('x3')
    mods.append(pym.MathGeneral(sxEr, sSIMP, expression="{0} + (1.0-{0})*inp0^3".format(xmin)))

    # Add stiffness assembly module
    sK = pym.Signal('K')
    mods.append(pym.AssembleGeneral(sSIMP, sK, domain=domain, element_matrix=el, bc=boundary_dofs))

    # Linear system solver
    su = pym.Signal('u')
    mods.append(pym.LinSolve([sK, sf], su))

    # Compliance calculation
    sc = pym.Signal('compliance')
    mods.append(pym.EinSum([su, sf], sc, expression='i,i->'))

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
    mods.append(ScaleTo(sc, sg0, val=100.0))

    # Plot design
    mods.append(pym.PlotDomain(sxNom, domain=domain, saveto="out/design"))
    # mods.append(pym.PlotDomain2D(sxEr, domain=domain))

    # Volume
    svol = pym.Signal('vol')
    mods.append(pym.EinSum(sxfilt, svol, expression='i->'))

    # Volume constraint
    sg1 = pym.Signal('volconstraint')
    mods.append(pym.MathGeneral(svol, sg1, expression='10*(inp0/{} - {})'.format(domain.nel, 0.5)))

    mods.append(pym.PlotIter([sg0]))

    # Compress all into a network
    func = pym.Network(mods)

    # --- OPTIMIZATION ---
    # pym.finite_difference(func, sx, sg0)
    pym.minimize_mma(func, [sx], [sg0, sg1])

