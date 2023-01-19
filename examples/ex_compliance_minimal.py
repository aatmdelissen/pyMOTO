""" Minimal example for a structural compliance topology optimization """
import pymodular as pym
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
    [3 / 2, -ka / 2 - kb / 2, kf, ka / 2 - kb, -kf, -ka + kb / 2, -3 / 2, ka + kb]]) # TODO somehow the x and y are mixed up
# OC update scheme
def oc_update(x, dfdx):
    l1, l2, move, maxvol = 0, 100000, 0.2, np.sum(x)
    while l2 - l1 > 1e-4:
        lmid = 0.5 * (l1 + l2)
        xnew = np.maximum(xmin, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x*np.sqrt(-dfdx/lmid)))))
        l1, l2 = (lmid, l2) if np.sum(xnew) - maxvol > 0 else (l1, lmid)
    return xnew, np.max(np.abs(xnew - x))
# Setup FE domain
domain = pym.DomainDefinition(nx, ny)  # Generate a grid
boundary_dofs = np.repeat(domain.get_nodenumber(0, np.arange(ny+1)) * 2, 2, axis=-1) + np.tile(np.arange(2), ny+1)  # Calculate boundary dof indices
f = np.zeros(domain.nnodes*2)  # Generate a force vector
f[2*domain.get_nodenumber(nx, ny//2)] = 1.0
# Initialize input signals
sf = pym.Signal('f', state=f)  # Force signal
sx = pym.Signal('x', state=np.ones(domain.nel)*volfrac)  # Design signal, with initial values
# Start building the modular network
func = pym.Network()
sxfilt = func.append(pym.Density(sx, domain=domain, radius=filter_radius))  # Filter
sSIMP  = func.append(pym.MathGeneral(sxfilt, expression=f"{xmin} + {1-xmin}*inp0^3"))  # SIMP material interpolation
sK     = func.append(pym.AssembleGeneral(sSIMP, domain=domain, element_matrix=el, bc=boundary_dofs))  # Add stiffness assembly module
su     = func.append(pym.LinSolve([sK, sf]))  # Linear system solver
sg0    = func.append(pym.EinSum([su, sf], expression='i,i->'))  # Compliance calculation
func.append(pym.PlotDomain(sxfilt, domain=domain, saveto="out/design"), pym.PlotIter([sg0]))  # Plot design and history
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
