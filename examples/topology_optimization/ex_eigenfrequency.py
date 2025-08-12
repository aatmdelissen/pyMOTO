""" 
Eigenfrequency maximization
===========================
Minimal example for an eigenfrequency topology optimization 

References:
- Du, J., Olhoff, N. (2007)
  Topological design of freely vibrating continuum structures for maximum values of simple and multiple eigenfrequencies and frequency gaps. 
  Structural and Multidisciplinary Optimization 34, 91-110 (2007). 
  DOI: https://doi.org/10.1007/s00158-007-0101-y
"""
import numpy as np
import pymoto as pym

nx, ny, nz = 50, 30, 0  # Set nz to zero for the 2D problem
xmin = 1e-9
filter_radius = 2.0
volfrac = 0.5

rho = 2700.0
E = 68.9e+9


class MassInterpolation(pym.Module):
    """ Two-range material interpolation based on Du and Olhoff's paper
    For x >= threshold:
        y = rho * x^p1
    For x < threshold:
        y = rho * x^p0 / (t^(p0-p1))
    """

    def __init__(self, rhoval=1.0, threshold=0.1, p0=6.0, p1=1.0):
        self.rhoval = rhoval
        self.threshold = threshold
        self.p0, self.p1 = p0, p1

    def __call__(self, x):
        xx = x ** self.p1
        xx[x < self.threshold] = x[x < self.threshold] ** self.p0 / (self.threshold ** (self.p0 - self.p1))
        return self.rhoval * xx

    def _sensitivity(self, drho):
        x = self.sig_in[0].state
        dx = self.p1 * x ** (self.p1 - 1)
        dx[x < self.threshold] = self.p0 * x[x < self.threshold]**(self.p0 - 1) / (self.threshold**(self.p0 - self.p1))
        return self.rhoval * dx * drho


if __name__ == "__main__":
    print(__doc__)

    if nz == 0:  # 2D structural eigenfrequency analysis
        # Generate a grid
        domain = pym.DomainDefinition(nx, ny)

        # Generate a non-design area that has mass
        nondesign_area = domain.elements[3*nx//4:, ny//4:ny*3//4].flatten()
    else:  # 3D
        # Generate a grid
        domain = pym.DomainDefinition(nx, ny, nz)

        # Generate a non-design area that has mass
        nondesign_area = domain.elements[3*nx//4:, ny//4:ny*3//4, nz//4:nz*3//4].flatten()

    # Calculate boundary dof indices
    boundary_nodes = domain.nodes[0, ...]
    boundary_dofs = domain.get_dofnumber(boundary_nodes, ndof=domain.dim)

    # Safety, to prevent overloading the memory in your machine; no problem to remove this
    if domain.nnodes > 1e+6:
        print("Too many nodes :(")  
        exit()

    # Make design vector, and fill with initial values
    sx = pym.Signal('x', state=np.ones(domain.nel) * volfrac)

    # Start building the modular network
    with pym.Network() as fn:
        # Density filter
        sxfilt = pym.DensityFilter(domain, radius=filter_radius)(sx)

        # Set the non-design domain
        sxndd = pym.VecSet(indices=nondesign_area, value=1.0)(sxfilt)

        sx_analysis = sxndd  # Alias for the design to perform the analysis on

        # Show the design on the screen as it optimizes
        pym.PlotDomain(domain, saveto="out/design", clim=[0, 1])(sx_analysis)

        # SIMP material interpolation
        # Note: Material properties can either be set in the scaling variables (sSIMP and sDENS; as is done here), or in
        # the assembly modules AssembleStiffness and AssembleMass by providing the relevant keyword arguments.
        sSIMP = pym.MathGeneral(f"{E}*({xmin} + {1.0 - xmin}*inp0^3)")(sx_analysis)
        sDENS = MassInterpolation(rhoval=rho)(sx_analysis)

        # Assemble mass and stiffness matrix
        sK = pym.AssembleStiffness(domain, bc=boundary_dofs)(sSIMP)
        sM = pym.AssembleMass(domain, bc=boundary_dofs, ndof=domain.dim)(sDENS)

        # Eigenvalue solver
        slams, seigvec = pym.EigenSolve(hermitian=True, nmodes=3)(sK, sM)
        slams.tag = "eigenvalues"
        seigvec.tag = "eigenvectors"

        # Output the design and eigenmodes to a Paraview file
        pym.WriteToVTI(domain, saveto='out/dat.vti')(sx_analysis, seigvec)

        # Get harmonic mean of three lowest eigenvalues
        sharm = pym.MathGeneral('1/inp0 + 1/inp1 + 1/inp2')(slams[0], slams[1], slams[2])
        sharm.tag = "harmonic mean"

        # MMA needs correct scaling of the objective
        sg0 = pym.Scaling(scaling=100.0)(sharm)
        sg0.tag = "objective"

        # Calculate the volume of the domain by adding all design densities together
        svol = pym.EinSum('i->')(sx_analysis)
        svol.tag = 'volume'

        # Volume constraint; note that also pym.Scaling(scaling=10.0, maxval=domain.nel*volfrac)(svol) could be used
        sg1 = pym.MathGeneral(f'10*(inp0/{domain.nel} - {volfrac})')(svol)
        sg1.tag = "volume constraint"

        # Maybe you want to check the design-sensitivities with finite difference?
        do_finite_difference = False
        if do_finite_difference:
            pym.finite_difference(sx, [sg0, sg1], dx=1e-4)
            exit()

        pym.PlotIter()(sg0, sg1)  # Plot iteration history
        pym.ScalarToFile('out/log.csv')(sharm, slams, svol)

    # Do the optimization with MMA
    pym.minimize_mma(sx, [sg0, sg1], verbosity=2)

    # Here you can do some post processing
    print("The optimization has finished!")
    print(f"The final eigenfrequencies obtained are {np.sqrt(slams.state)}")
