""" Minimal example for a structural compliance topology optimization """
import pyModular as pym
import numpy as np
nx, ny = 100, 40
xmin, filter_radius, volfrac = 1e-3, 1.5, 0.5
# Compute element stiffness matrix
nu, E = 0.3, 1.0
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
# OC update scheme
def oc_update(x, dfdx):
    l1, l2, move = 0, 100000, 0.2
    while l2 - l1 > 1e-4:
        lmid = 0.5 * (l1 + l2)
        xnew = np.maximum(xmin, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x*np.sqrt(-dfdx/lmid)))))
        if np.sum(xnew) - volfrac*nx*ny > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew, np.max(np.abs(xnew - x))
# Setup FE domain
domain = pym.DomainDefinition(nx, ny)  # Generate a grid
boundary_dofs = np.repeat(domain.get_nodenumber(0, np.arange(ny+1)) * 2, 2, axis=-1) + np.tile(np.arange(2), ny+1)  # Calculate boundary dof indices
f = np.zeros(domain.nnodes*2)  # Generate a force vector
f[2*domain.get_nodenumber(nx, ny//2)] = 1.0
# Initialize input signals
sf = pym.Signal('f', state=f)  # Force signal
sx = pym.Signal('x', state=np.ones(domain.nel)*volfrac)  # Design signal, with initial values
sxfilt, sSIMP, sK, su, sg0 = pym.make_signals('xfiltered', 'x3', 'K', 'u', 'compliance')
# Start building the modular network
func = pym.Network()
func.append(pym.Density(sx, sxfilt, domain=domain, radius=filter_radius))  # Filter
func.append(pym.MathGeneral(sxfilt, sSIMP, expression="{0} + (1.0-{0})*inp0^3".format(xmin)))  # SIMP material interpolation
func.append(pym.AssembleGeneral(sSIMP, sK, domain=domain, element_matrix=el, bc=boundary_dofs))  # Add stiffness assembly module
func.append(pym.LinSolve([sK, sf], su))  # Linear system solver
func.append(pym.EinSum([su, sf], sg0, expression='i,i->'))  # Compliance calculation
func.append(pym.PlotDomain2D(sxfilt, domain=domain, saveto="out/design"), pym.PlotIter([sg0]))  # Plot design and history
# Perform the actual optimization, using OC
loop, change = 0, 1.0
while change > 0.01:
    loop += 1
    func.response()        # Forward analysis
    func.reset()           # Clear previous sensitivities
    sg0.sensitivity = 1.0  # Sensitivity seed
    func.sensitivity()     # Backpropagation
    sx.state, change = oc_update(sx.state, sx.sensitivity)
    print("It {0: 3d}, g0 {1:.3e}, vol {2:.3f}, change {3:.2f}".format(loop, sg0.state, np.sum(sx.state)/(nx*ny), change))
