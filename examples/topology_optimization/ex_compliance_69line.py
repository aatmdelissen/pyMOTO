"""PyMOTO 69-line topology optimization example
===============================================
Example for a structural compliance topology optimization based on the 99 line Matlab script of Sigmund.
Using pyMOTO the number of lines can be reduced to only 69 lines (while providing the same level of detail), which
even includes blank lines for readability and additional comments to clarify the functionality of this example.
Reference:
    Sigmund (2001), "A 99 line topology optimization code written in Matlab" (https://doi.org/10.1007/s001580050176)
"""
import numpy as np
import pymoto as pym

nx, ny = 100, 40  # Size of the domain (number of elements)
xmin, filter_radius, volfrac = 1e-3, 1.5, 0.5  # Minimum density, density-filter radius, and volume fraction

# Compute element stiffness matrix
nu, E = 0.3, 1.0
ka, kb, kf = 4 * (1 - nu), 2 * (1 - 2 * nu), 6 * nu - 3 / 2
el = (E/ (12 * (1 + nu) * (1 - 2 * nu)) * np.array([
    [ka + kb, -3 / 2, ka / 2 - kb, kf, -ka + kb / 2, -kf, -ka / 2 - kb / 2, 3 / 2],
    [-3 / 2, ka + kb, -kf, -ka + kb / 2, kf, ka / 2 - kb, 3 / 2, -ka / 2 - kb / 2],
    [ka / 2 - kb, -kf, ka + kb, 3 / 2, -ka / 2 - kb / 2, -3 / 2, -ka + kb / 2, kf],
    [kf, -ka + kb / 2, 3 / 2, ka + kb, -3 / 2, -ka / 2 - kb / 2, -kf, ka / 2 - kb],
    [-ka + kb / 2, kf, -ka / 2 - kb / 2, -3 / 2, ka + kb, 3 / 2, ka / 2 - kb, -kf],
    [-kf, ka / 2 - kb, -3 / 2, -ka / 2 - kb / 2, 3 / 2, ka + kb, kf, -ka + kb / 2],
    [-ka / 2 - kb / 2, 3 / 2, -ka + kb / 2, -kf, ka / 2 - kb, kf, ka + kb, -3 / 2],
    [3 / 2, -ka / 2 - kb / 2, kf, ka / 2 - kb, -kf, -ka + kb / 2, -3 / 2, ka + kb]]))


# OC update scheme
def oc_update(x, dfdx):
    l1, l2, move, maxvol = 0, 100000, 0.2, np.sum(x)
    while l2 - l1 > 1e-4:
        lmid = 0.5 * (l1 + l2)
        xnew = np.clip(x * np.sqrt(-dfdx / lmid), np.maximum(0, x - move), np.minimum(1, x + move))
        l1, l2 = (lmid, l2) if np.sum(xnew) > maxvol else (l1, lmid)
    return xnew, np.max(np.abs(xnew - x))


# Setup FE domain
domain = pym.VoxelDomain(nx, ny)  # Generate a discretization grid
n_left = domain.nodes[0, :].flatten()  # Get node number of all nodes on the left side (x=0)
boundary_dofs = np.concatenate([n_left * 2, n_left * 2 + 1])  # Calculate boundary dof indices to fix
f = np.zeros(domain.nnodes * 2)  # Generate a force vector
f[2 * domain.nodes[-1, ny // 2]] = 1.0  # Set a force in the middle of the right side

# Initialize input signal for design variables
sx = pym.Signal("x", state=np.ones(domain.nel) * volfrac)

# Start building the modular network; this constructs a function of which we can calculate sensitivities easily
with pym.Network() as func:
    sxfilt = pym.DensityFilter(domain=domain, radius=filter_radius)(sx)  # Density filter
    sSIMP = pym.MathExpression(expression=f"{xmin} + {1 - xmin}*inp0^3")(sxfilt)  # SIMP material interpolation
    sK = pym.AssembleGeneral(domain=domain, element_matrix=el, bc=boundary_dofs)(sSIMP)  # Stiffness matrix assembly
    su = pym.LinSolve()(sK, f)  # Solver for linear system of equations
    sg0 = pym.EinSum(expression="i,i->")(su, f)  # Compliance calculation
    pym.PlotDomain(domain=domain, saveto="out/design")(sxfilt), pym.PlotIter()(sg0)  # Show design and iter history

# Perform the actual optimization, using OC
print("Initial g0 {0:.3e}, vol {1:.3f}".format(sg0.state, np.sum(sx.state) / (nx * ny)))
loop, change = 0, 1.0
while change > 0.01:
    loop += 1
    func.reset()  # Clear previous sensitivities
    sg0.sensitivity = 1.0  # Seed the last sensitivity with a value of 1.0
    func.sensitivity()  # Backpropagation to calculate all sensitivities
    sx.state, change = oc_update(sx.state, sx.sensitivity)
    func.response()  # Forward analysis (note: the first is already calculated so only done after the first update)
    vol = np.sum(sx.state) / (nx * ny)
    print("It {0: 3d}, g0 {1:.3e}, vol {2:.3f}, change {3:.2f}".format(loop, sg0.state, vol, change))
