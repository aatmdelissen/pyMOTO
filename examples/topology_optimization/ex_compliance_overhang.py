"""Overhang filter
==================

In this example the usage of an overhang filter is demonstrated to obtain a printable design

The module :py:class:`pymoto.OverhangFilter` removes all overhanging material, which forces the optimizer to add 
supports to load-bearing parts of the design. Compared to the example 
:ref:`sphx_glr_auto_examples_topology_optimization_ex_compliance.py`, only the overhang filter is added (see line 64-65 
of this example).

References:
  - Langelaar, M. (2017). *An additive manufacturing filter for topology optimization of print-ready designs*.
    Structural and Multidisciplinary Optimization, 55(3), 871-883.
    `doi: 10.1007/s00158-016-1522-2 <https://doi.org/10.1007/s00158-016-1522-2>`_
  - Langelaar, M. (2016). *Topology optimization of 3D self-supporting structures for additive manufacturing*.
    Additive Manufacturing, 12, 60-70.
    `doi: 10.1016/j.addma.2016.06.010 <https://doi.org/10.1016/j.addma.2016.06.010>`_
  - Delissen, A. *et al.* (2022). *Realization and assessment of metal additive manufacturing and topology optimization 
    for high-precision motion systems*. Additive Manufacturing, 58, 103012.
    `doi: 10.1016/j.addma.2022.103012 <https://doi.org/10.1016/j.addma.2022.103012>`_
"""
import numpy as np

import pymoto as pym

nx, ny, nz = 120, 40, 0  # Set nz to zero for the 2D problem, nz > 0 runs a 3D problem
xmin = 1e-9
filter_radius = 2.0
volfrac = 0.5

if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D analysis
        # Generate a grid
        domain = pym.VoxelDomain(nx, ny)
        ndof = domain.dim
        # Calculate boundary dof indices
        boundary_dofs = domain.get_dofnumber(domain.nodes[0, :], np.arange(ndof), ndof=ndof).flatten() 

        # Which dofs to put a force on? The 1 is added for a force in y-direction (x-direction would be zero)
        force_dofs = ndof * domain.nodes[nx, int(ny/2)] + 1

    else:
        domain = pym.VoxelDomain(nx, ny, nz)
        ndof = domain.dim
        boundary_dofs = domain.get_dofnumber(domain.nodes[0, :, :], np.arange(ndof), ndof=ndof).flatten()  
        force_dofs = ndof * domain.nodes[nx, :, int(nz/2)] + 2  # Z-direction

    # Generate a force vector
    f = np.zeros(domain.nnodes * ndof)
    f[force_dofs] = 1.0  # Uniform force of 1.0 at all selected dofs

    # Make signal for design vector, and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)
    
    # Start building the modular network
    # Filter
    sxfilt = pym.FilterConv(domain, radius=filter_radius, ymin_bc=0, ymax_bc=0, zmin_bc=0, zmax_bc=0)(sx)
    sxfilt.tag = "Filtered design"

    # -----> Only the :py:class:`pymoto.OverhangFilter` module needs to be added after filtering. The rest of the 
    # example is equal to :ref:`sphx_glr_auto_examples_topology_optimization_ex_compliance.py` 
    sxoverhang = pym.OverhangFilter(domain, direction='+y')(sxfilt)
    sxoverhang.tag = "Printable design"
    # <-----

    sx_analysis = sxoverhang

    # Show the design on the screen as it optimizes
    if domain.dim == 2:
        pym.PlotDomain(domain, saveto="out/design", clim=[0, 1])(sx_analysis)

    # SIMP material interpolation
    sSIMP = pym.MathExpression(f"{xmin} + {1.0 - xmin}*inp0^3")(sx_analysis)

    # System matrix assembly module
    sK = pym.AssembleStiffness(domain, bc=boundary_dofs)(sSIMP)

    # Linear system solver
    su = pym.LinSolve(symmetric=True, positive_definite=True)(sK, f)

    # # Output the design, deformation, and force field to a Paraview file
    pym.WriteToVTI(domain, saveto='out/dat.vti')(sx_analysis, su, f)

    # Compliance calculation c = f^T u
    scompl = pym.EinSum('i, i->')(su, f)
    scompl.tag = 'compliance'

    # MMA needs correct scaling of the objective
    sg0 = pym.Scaling(scaling=100.0)(scompl)
    sg0.tag = "objective"

    # Calculate the volume of the domain by adding all design densities together
    svol = pym.EinSum('i->')(sx_analysis)
    svol.tag = 'volume'

    # Volume constraint
    sg1 = pym.MathExpression(f'10*(inp0/{domain.nel} - {volfrac})')(svol)
    sg1.tag = "volume constraint"

    # Maybe you want to check the design-sensitivities?
    do_finite_difference = False
    if do_finite_difference:
        pym.finite_difference(sx, [sg0, sg1], dx=1e-4)
        exit()

    pym.PlotIter()(sg0, sg1)  # Plot iteration history

    # Do the optimization with MMA
    pym.minimize_mma(sx, [sg0, sg1])

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final compliance value obtained is {scompl.state}")
    print(f"The maximum displacement is {max(np.absolute(su.state))}")
