""" 
Topology optimization with robust formulation
=============================================

The robust formulation ensures that the design is tolerant for manufacturing deviations. Also it allows for control 
of the minimum feature size in the design. It is implemented using Heaviside projections on the design, generating
three variants: a nominal design, an eroded design and a dilated design. The worst-case of these three designs is used
in the objective function. Note that for a compliance minimization problem, the eroded design is always the worst. 
Hence it is called the 'poor-man's robust formuation'.

References:
- Wang, F., Lazarov, B. S., & Sigmund, O. (2011).
  On projection methods, convergence and robust formulations in topology optimization.
  Structural and multidisciplinary optimization, 43, 767-784.
  DOI: https://doi.org/10.1007/s00158-010-0602-y
"""
import numpy as np
import pymoto as pym

nx, ny = 100, 40
xmin = 1e-5
filter_radius = 5.0
volfrac = 0.5


class Continuation(pym.Module):
    """ Module that generates a continuated value """

    def __init__(self, start=0.0, stop=1.0, nsteps=80, stepstart=10, interval=10):
        self.startval = start
        self.endval = stop
        self.interval = interval
        self.dval = (stop - start) / nsteps
        self.nstart = stepstart
        self.iter = -1
        self.val = self.startval

    def __call__(self):
        maxval = max(self.startval, self.endval)
        minval = min(self.startval, self.endval)
        if self.iter % self.interval == 0:  # Only update value every `interval` iterations
            self.val = np.clip(self.startval + self.dval * (self.iter - self.nstart), minval, maxval)
        
        self.iter += 1
        return self.val


if __name__ == "__main__":
    print(__doc__)

    # --- SETUP ---
    # Generate a grid
    domain = pym.DomainDefinition(nx, ny)

    # Chose which physics to solve: structural or thermal
    physics = "structural"  # "strucural" or "thermal" 
    # (the thermal problem with robust is a lot harder for the optimizer)
    if physics == "structural":
        # STRUCTURAL
        # Calculate boundary dof indices
        boundary_nodes = domain.get_nodenumber(0, np.arange(ny + 1))
        boundary_dofs = domain.get_dofnumber(boundary_nodes, np.arange(2), 2).flatten()

        # Generate a force vector
        force_dofs = 2 * domain.get_nodenumber(nx, ny // 2)
        ndof = 2  # Number of displacements per node

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
        boundary_dofs = domain.get_nodenumber(0, np.arange(ny // 4, (3 * ny) // 4))

        # Make a force vector
        force_dofs = domain.get_nodenumber(*np.meshgrid(np.arange(nx // 4, nx + 1), np.arange(ny + 1)))
        ndof = 1  # Number of dofs per node

        # Element conductivity matrix
        el = 1 / 6 * np.array([
            [8, -2, -2, -4],
            [-2, 8, -4, -2],
            [-2, -4, 8, -2],
            [-4, -2, -2, 8]
        ])
    else:
        raise RuntimeError("Unknown physics: {}".format(physics))

    # In this example the filter is (virtually) padded with zeros, around the boundary condition we don't want that to 
    # happen. This vector sets the elements for which no padding is applied.
    pad = domain.elements[:int(filter_radius)+1, :].flatten()

    # Make force and design vector, and fill with initial values
    f = np.zeros(domain.nnodes * ndof)
    f[force_dofs] = 1.0

    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    with pym.Network() as fn:
        # Filter
        sxfilt = pym.DensityFilter(domain=domain, radius=filter_radius, nonpadding=pad)(sx)
        sxfilt.tag = "filtered design"

        # Heaviside projections
        sBeta = Continuation(start=1e-3, stop=40.0, stepstart=10, nsteps=80, interval=5)()
        sBeta.tag = "beta"

        pym.Print()(sBeta)  # Print the beta value each iteration

        etaDi, etaNo, etaEr = 0.4, 0.5, 0.6

        heaviside = "(tanh(inp1 * {0}) + tanh(inp1 * (inp0 - {0}))) / (tanh(inp1 * {0}) + tanh(inp1 * (1 - {0})))"
        sxNom = pym.MathGeneral(heaviside.format(etaNo))(sxfilt, sBeta)
        sxEr = pym.MathGeneral(heaviside.format(etaEr))(sxfilt, sBeta)
        sxDi = pym.MathGeneral(heaviside.format(etaDi))(sxfilt, sBeta)

        sxNom.tag = "nominal"
        sxEr.tag = "eroded"
        sxDi.tag = "dilated"

        # SIMP material interpolation
        sSIMP = pym.MathGeneral(f"{xmin} + {1-xmin}*inp0^3")(sxEr)

        # Add stiffness assembly module
        sK = pym.AssembleGeneral(domain, element_matrix=el, bc=boundary_dofs)(sSIMP)

        # Linear system solver
        su = pym.LinSolve()(sK, f)

        # Compliance calculation
        sc = pym.EinSum('i,i->')(su, f)

        # Objective scaling
        sg0 = pym.Scaling(scaling=100.0)(sc)
        sg0.tag = "objective"

        # Plot design
        pym.PlotDomain(domain, saveto="out/design")(sxNom)

        # Volume
        s_volume = pym.EinSum('i->')(sxNom)

        # Volume constraint
        s_gvol = pym.Scaling(scaling=10, maxval=volfrac * domain.nel)(s_volume)
        s_gvol.tag = "Volume constraint"
        
        pym.PlotIter()(sg0)

    # --- OPTIMIZATION ---
    # pym.finite_difference(func, sx, sg0)  # Note that the FD won't be correct due to the continuation changing values
    pym.minimize_mma(sx, [sg0, s_gvol], maxit=120)

