""" Minimal example for a structural compliance topology optimization """
import pymodular as pym
import numpy as np
# import sao
import time

nx, ny, nz = 40, 40, 40
xmin = 1e-9
filter_radius = 4.0
volfrac = 0.5
move = 0.2

def oc_update(x, dfdx):
    l1, l2 = 0, 100000
    xU = np.minimum(x + move, 1.0)
    xL = np.maximum(x - move, xmin)
    c = x * np.sqrt(-dfdx)
    vmax = volfrac*nx*ny
    while (l2 - l1)/(l1 + l2) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        xnew = np.clip(c/np.sqrt(lmid), xL, xU)
        if np.sum(xnew) > vmax:
            l1 = lmid
        else:
            l2 = lmid
    return xnew, np.max(np.abs(xnew - x))

if __name__ == "__main__":
    print(__doc__)
    start = time.time()

    if nz == 0:  # 2D analysis
        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        # Calculate boundary dof indices
        boundary_nodes = domain.get_nodenumber(0, np.arange(ny+1))
        boundary_dofs = np.repeat(boundary_nodes * 2, 2, axis=-1) + np.tile(np.arange(2), len(boundary_nodes))

        force_dofs = domain.dim*domain.get_nodenumber(nx, ny//2) + 1  # y-direction
    else:
        domain = pym.DomainDefinition(nx, ny, ny)

        boundary_nodes = domain.get_nodenumber(*np.meshgrid(0, range(ny+1), range(ny+1))).flatten()
        boundary_dofs = np.repeat(boundary_nodes * 3, 3, axis=-1) + np.tile(np.arange(3), len(boundary_nodes))

        force_dofs = domain.dim*domain.get_nodenumber(nx, ny//2, ny//2)+2  # Z-direction
    if domain.nnodes > 1e+6:
        print("Too many nodes :(")
        exit()

    # Generate a force vector
    f = np.zeros(domain.nnodes*domain.dim)
    f[force_dofs] = 1.0

    # Make force and design vector, and fill with initial values
    sf = pym.Signal('f', state=f)
    sx = pym.Signal('x', state=np.ones(domain.nel)*volfrac)

    # Start building the modular network
    func = pym.Network()

    # Filter
    sxfilt = func.append(pym.Density(sx, domain=domain, radius=filter_radius))
    # ep = 0.1
    # sxfilt = func.append(pym.FilterConv(sx, domain=domain, mode='wrap', weights=np.array([[0, ep, 0], [ep, 1, ep], [0, ep, 0]])/(1+4*ep)))
    # func[-1].set_filter_radius(10.0, element_units=True)
    # sxprint1 = func.append(pym.OverhangFilter(sxfilt, domain=domain, direction='y'))
    # sxprint = func.append(pym.OverhangFilter(sxprint1, domain=domain, direction='y-'))

    # Printability constraint
    # sxdiff = func.append(pym.MathGeneral([sxfilt, sxprint], expression=f"(inp0-inp1)^2"))
    # sVdiff = func.append(pym.EinSum(sxdiff, expression='i->'))
    # sg2 = func.append(pym.MathGeneral(sVdiff, expression=f"10*((inp0/{domain.nel})/0.01 - 1.0)"))

    sx_analysis = sxfilt
    # SIMP material interpolation
    sSIMP = func.append(pym.MathGeneral(sx_analysis, expression=f"{xmin} + {1.0-xmin}*inp0^3"))

    # Add stiffness assembly module
    sK = func.append(pym.AssembleStiffness(sSIMP, domain=domain, bc=boundary_dofs))
    # sK = func.append(pym.AssembleGeneral(sSIMP, domain=domain, element_matrix=el, bc=boundary_dofs))

    # Linear system solver
    # solver = pym.SolverSparsePardiso()
    # solver = pym.SolverSparseCholeskyCVXOPT()
    # solver = pym.SolverSparseCholeskyScikit()
    su = func.append(pym.LinSolve([sK, sf], hermitian=True, solver=None))

    func.append(pym.WriteToParaview([sx_analysis, su, sf], domain=domain, saveto='out/dat.vti'))

    # Compliance calculation
    sg0 = func.append(pym.EinSum([su, sf], expression='i,i->'))

    # Plot some information
    # func.append(pym.PlotDomain2D(sxprint, domain=domain, saveto="out/design", clim=[0, 1]))  # Plot design
    func.append(pym.PlotDomain2D(sx_analysis, domain=domain, saveto="out/design", clim=[0, 1]))  # Plot design
    # func.append(pym.PlotIter([sg0]))  # Plot iteration history

    # Volume constraint
    svol = func.append(pym.EinSum(sx_analysis, expression='i->'))
    sg1 = func.append(pym.MathGeneral(svol, expression='10*(inp0/{} - {})'.format(domain.nel, volfrac)))
    sg1.tag = "volume constraint"

    # pym.finite_difference(func, sx, [sg0], dx=1e-4)
    # exit()

    # subprob = sao.problems.Subproblem(
    #     approximation=sao.approximations.Taylor1(intervening=sao.intervening_variables.mma.MMA02(x_min=0, x_max=1)),
    #     limits=[sao.move_limits.Bounds(xmin=0., xmax=1.), sao.move_limits.MoveLimit(move_limit=0.2)])

    df = np.zeros((2, sx.state.size))
    f = np.zeros(2)
    print(f"Setup in {time.time()-start} s")
    max_iter = 50
    t_solver = np.zeros(max_iter)
    for loop in range(max_iter):
        itstart = time.time()
        # func.response()
        for md in func.mods:
            start = time.time()
            md.response()        # Forward analysis
            dur = time.time()-start
            print(f"Response of {type(md).__name__} calculated in {dur} s")
            if isinstance(md, pym.LinSolve):
                t_solver[loop] = dur
        # f[0] = sg0.state
        # f[1] = sg1.state
        # f[2] = sg2.state

        start = time.time()
        func.reset()           # Clear previous sensitivities
        sg0.sensitivity = 1.0  # Sensitivity seed
        func.sensitivity()     # Backpropagation
        # df[0, :] = sx.sensitivity
        print(f"Sensitivity in {time.time()-start} s")

        # func.reset()           # Clear previous sensitivities
        # sg1.sensitivity = 1.0  # Sensitivity seed
        # func.sensitivity()     # Backpropagation
        # df[1, :] = sx.sensitivity

        # func.reset()           # Clear previous sensitivities
        # sg2.sensitivity = 1.0  # Sensitivity seed
        # func.sensitivity()     # Backpropagation
        # df[2, :] = sx.sensitivity

        # Build approximate sub-problem at X^(k)
        # subprob.build(sx.state, f, df)

        # Solve current subproblem
        start = time.time()
        # sx.state[:] = sao.solvers.pdip(subprob)[0]
        # sx.state[:] = sao.solvers.ipsolver(subprob)
        sx.state[:] = oc_update(sx.state, sx.sensitivity)[0]
        # sx.state[:] = sao.solvers.scipy_solver(subprob)
        print(f"New design in {time.time()-start} s")

        print("It {0: 3d}, g0 {1:.3e}, vol {2:.3f}".format(loop, sg0.state, np.sum(sx.state)/(nx*ny)))
        print(f"Iteration finished in {time.time()-itstart} s")

    [print(t) for t in t_solver]


    # loop = 0
    # change = 1.0
    # while change > 0.01:
    #     loop += 1
    #     func.response()        # Forward analysis
    #     func.reset()           # Clear previous sensitivities
    #     sg0.sensitivity = 1.0  # Sensitivity seed
    #     func.sensitivity()     # Backpropagation
    #
    #     # data = sx.sensitivity.reshape((domain.nx, domain.ny), order='F').T
    #     # if im is None:
    #     #     im = ax.imshow(data, origin='lower', cmap='seismic')
    #     #     cbar = fig.colorbar(im, orientation='horizontal')
    #     # else:
    #     #     im.set_data(data)
    #     # mx = max(abs(np.min(data)), abs(np.max(data)))
    #     # im.set_clim(vmin=-mx, vmax=mx)
    #     # fig.canvas.draw()
    #     # fig.canvas.flush_events()
    #     sx.state, change = oc_update(sx.state, sx.sensitivity)
    #
    #     print("It {0: 3d}, g0 {1:.3e}, vol {2:.3f}, change {3:.2f}".format(loop, sg0.state, np.sum(sx.state)/(nx*ny), change))


