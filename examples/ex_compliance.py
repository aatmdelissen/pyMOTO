""" Minimal example for a structural compliance topology optimization """
import pymodular as pym
import numpy as np

nx, ny = 100, 40
xmin = 1e-3
filter_radius = 1.5
volfrac = 0.5


def oc_update(x, dfdx):
    l1, l2, move = 0, 100000, 0.2
    while l2 - l1 > 1e-4:
        lmid = 0.5 * (l1 + l2)
        xnew = np.maximum(xmin, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x*np.sqrt(-dfdx/lmid)))))
        if np.sum(xnew) - volfrac*nx*ny > 0:
            l1 = lmid
        else:
            l2 = lmid
    change = np.max(np.abs(xnew - x))
    return xnew, change


if __name__ == "__main__":
    print(__doc__)

    # Generate a grid
    domain = pym.DomainDefinition(nx, ny)

    # Calculate boundary dof indices
    boundary_nodes = domain.get_nodenumber(0, np.arange(ny+1))
    boundary_dofs = np.repeat(boundary_nodes * 2, 2, axis=-1) + np.tile(np.arange(2), len(boundary_nodes))

    # Generate a force vector
    force_dofs = 2*domain.get_nodenumber(nx, ny//2)
    f = np.zeros(domain.nnodes*2)
    f[force_dofs] = 1.0

    # Compute element stiffness matrix
    nu, E = 0.3, 1.0
    lx, ly, lz = 1.0, 1.0, 1.0
    c = ly / lx
    ka, kc = (4 * c) * (1 - nu), (4 / c) * (1 - nu)
    kd, kb = (2 * c) * (1 - 2 * nu), (2 / c) * (1 - 2 * nu)
    kf = 6 * nu - 3 / 2

    el = E * lz / (12 * (1 + nu) * (1 - 2 * nu)) * np.array([
        [ka + kb, -3 / 2, ka / 2 - kb, kf, -ka + kb / 2, -kf, -ka / 2 - kb / 2, 3 / 2],
        [-3 / 2, kc + kd, -kf, -kc + kd / 2, kf, kc / 2 - kd, 3 / 2, -kc / 2 - kd / 2],
        [ka / 2 - kb, -kf, ka + kb, 3 / 2, -ka / 2 - kb / 2, -3 / 2, -ka + kb / 2, kf],
        [kf, -kc + kd / 2, 3 / 2, kc + kd, -3 / 2, -kc / 2 - kd / 2, -kf, kc / 2 - kd],
        [-ka + kb / 2, kf, -ka / 2 - kb / 2, -3 / 2, ka + kb, 3 / 2, ka / 2 - kb, -kf],
        [-kf, kc / 2 - kd, -3 / 2, -kc / 2 - kd / 2, 3 / 2, kc + kd, kf, -kc + kd / 2],
        [-ka / 2 - kb / 2, 3 / 2, -ka + kb / 2, -kf, ka / 2 - kb, kf, ka + kb, -3 / 2],
        [3 / 2, -kc / 2 - kd / 2, kf, kc / 2 - kd, -kf, -kc + kd / 2, -3 / 2, kc + kd]])

    # Make force and design vector, and fill with initial values
    sf = pym.Signal('f', state=f)
    sx = pym.Signal('x', state=np.ones(domain.nel)*volfrac)

    # Start building the modular network
    func = pym.Network()

    # Filter
    sxfilt = pym.Signal('xfiltered')
    func.append(pym.Density(sx, sxfilt, domain=domain, radius=filter_radius))

    # SIMP material interpolation
    sSIMP = pym.Signal('x3')
    func.append(pym.MathGeneral(sxfilt, sSIMP, expression="{0} + (1.0-{0})*inp0^3".format(xmin)))

    # Add stiffness assembly module
    sK = pym.Signal('K')
    func.append(pym.AssembleGeneral(sSIMP, sK, domain=domain, element_matrix=el, bc=boundary_dofs))

    # Linear system solver
    su = pym.Signal('u')
    func.append(pym.LinSolve([sK, sf], su))

    # Compliance calculation
    sg0 = pym.Signal('compliance')
    func.append(pym.EinSum([su, sf], sg0, expression='i,i->'))

    # Plot some information
    func.append(pym.PlotDomain2D(sxfilt, domain=domain, saveto="out/design"))  # Plot design
    func.append(pym.PlotIter([sg0]))  # Plot iteration history

    # Perform the actual optimization, using OC
    loop = 0
    change = 1.0
    while change > 0.01:
        loop += 1
        func.response()        # Forward analysis
        func.reset()           # Clear previous sensitivities
        sg0.sensitivity = 1.0  # Sensitivity seed
        func.sensitivity()     # Backpropagation
        sx.state, change = oc_update(sx.state, sx.sensitivity)
        print("It {0: 3d}, g0 {1:.3e}, vol {2:.3f}, change {3:.2f}".format(loop, sg0.state, np.sum(sx.state)/(nx*ny), change))


