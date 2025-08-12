"""
Flexure design
==============
Example of the design of a flexure using topology optimization with:
(i) maximum shear stiffness
(ii) constrainted maximum stiffness in axial stiffness
(iii) (optional) constrained maximum use of material (volume constraint)

References:
- Koppen, S., Langelaar, M., & van Keulen, F. (2022).
  A simple and versatile topology optimization formulation for flexure synthesis.
  Mechanism and Machine Theory, 172, 104743.
  DOI: http://dx.doi.org/10.1016/j.mechmachtheory.2022.104743
"""
import numpy as np
import pymoto as pym


class Symmetry(pym.Module):
    """Enforce symmetry in the x and y direction """
    def __init__(self, domain: pym.DomainDefinition):
        self.domain = domain

    def __call__(self, x):
        x = np.reshape(x, (self.domain.nely, self.domain.nelx))
        x = (x + np.flip(x, 0)) / 2
        x = (x + np.flip(x, 1)) / 2
        return x.flatten()

    def _sensitivity(self, dfdv):
        dfdv = np.reshape(dfdv, (self.domain.nely, self.domain.nelx))
        dfdv = (dfdv + np.flip(dfdv, 1)) / 2
        dfdv = (dfdv + np.flip(dfdv, 0)) / 2
        return dfdv


def flexure(nx: int = 20, 
            ny: int = 20, 
            doc: str = 'tx', 
            dof: str = 'ty', 
            emax: float = 1.0,
            filter_radius: float = 2.0, 
            E: float = 100.0, 
            nu: float = 0.3, 
            xmin: float = 1e-9,
            volfrac=0.3, 
            initial_volfrac: float = 0.2, 
            use_symmetry=False):
    """Run a flexure optimization
    The compliance of the mechanism mode (`dof`) is maximized and the constraint mode is constrained (`doc`).

    Args:
        nx (int, optional): Number of elements in x-direction. Defaults to 20.
        ny (int, optional): Number of elements in y-direction. Defaults to 20.
        doc (str, optional): Direction of constraint. Defaults to 'tx'.
        dof (str, optional): Direction of freedom (mechanism mode). Defaults to 'ty'.
        emax (float, optional): Maximum compliance value. Defaults to 1.0.
        filter_radius (float, optional): Filter radius. Defaults to 2.0.
        E (float, optional): Young's modulus. Defaults to 100.0.
        nu (float, optional): Poisson ratio. Defaults to 0.3.
        xmin (float, optional): Minimum pseudo-density value. Defaults to 1e-9.
        volfrac (float, optional): Volume fraction. Defaults to 0.3.
        initial_volfrac (float, optional): Volume fraction in the initial design. Defaults to 0.2.
        use_symmetry (bool, optional): Enforce symmetry in the design. Defaults to False.
    """
    

    # Set up the domain
    domain = pym.DomainDefinition(nx, ny)

    # Node and dof groups
    nodes_top = domain.nodes[:, -1]
    nodes_bottom = domain.nodes[:, 0]

    dofs_top = domain.get_dofnumber(nodes_top, [0, 1], ndof=2)
    dofs_bottom = domain.get_dofnumber(nodes_bottom, [0, 1], ndof=2)

    all_dofs = np.arange(0, 2 * domain.nnodes)
    prescribed_dofs = np.unique(np.hstack([dofs_bottom.flatten(), dofs_top.flatten()]))
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

    dofs_top_x = dofs_top[..., 0].flatten()
    dofs_top_y = dofs_top[..., 1].flatten()

    # Construct deformation modes 
    v = np.zeros((2 * domain.nnodes, 3), dtype=float)
    v[dofs_top_x, 0] = 1.0  # tx
    v[dofs_top_y, 1] = 1.0  # ty
    v[dofs_top_x, 2] = 1.0 * ny / nx  # rz
    v[dofs_top_y, 2] = np.linspace(1, -1, nx + 1)
    
    # Select directions for constraint/mechanism
    degrees = ('tx', 'ty', 'rz')
    u = v[:, [degrees.index(doc), degrees.index(dof)]]

    # Signal with design variables
    s_x = pym.Signal('x', state=initial_volfrac * np.ones(domain.nel))

    # Setup optimization problem
    with pym.Network() as fn:
        # Force symmetry
        if use_symmetry:
            s_x1 = Symmetry(domain)(s_x)
        else:
            s_x1 = s_x
            
        # Density filtering
        s_xfilt = pym.DensityFilter(domain, radius=filter_radius)(s_x1)
        s_xfilt.tag = 'Filtered density'

        # Plot the design
        pym.PlotDomain(domain, saveto="out/design")(s_xfilt)

        # SIMP penalization
        s_xsimp = pym.MathGeneral(f"{xmin} + {1 - xmin}*inp0^3")(s_xfilt)

        # Assembly of stiffness matrix
        s_K = pym.AssembleStiffness(domain, e_modulus=E, poisson_ratio=nu)(s_xsimp)

        # Solve system of equations for the two loadcases
        up = u[prescribed_dofs, :]
        ff = np.zeros((free_dofs.size, 2))
        s_u, s_f = pym.SystemOfEquations(free=free_dofs, prescribed=prescribed_dofs)(s_K, ff, up)

        # Calculate compliances for each loadcase [constraint, mechanism]
        s_compl = pym.EinSum('ij,ij->j')(s_u, s_f)

        # Objective function: maximize compliance of mechanism mode
        s_objective = pym.Scaling(scaling=-10)(s_compl[0])
        s_objective.tag = "Objective"

        # Compliance constraint on the constraint mode
        s_compliance_constraint = pym.Scaling(scaling=10, maxval=emax)(s_compl[1])
        s_compliance_constraint.tag = "Compliance constraint"

        # List of optimization responses: first the objective and all others the constraints
        responses = [s_objective, s_compliance_constraint]

        # Add volume constraint if requested
        if volfrac:
            # Volume
            s_volume = pym.EinSum('i->')(s_xfilt)

            # Volume constraint
            s_volume_constraint = pym.Scaling(scaling=10, maxval=volfrac * domain.nel)(s_volume)
            s_volume_constraint.tag = "Volume constraint"

            responses.append(s_volume_constraint)
   
        pym.PlotIter()(*responses)

    # Optimization
    pym.minimize_mma(s_x, responses, verbosity=2, maxit=200)


if __name__ == "__main__":
    # flexure(100, 120, 'rz', 'ty', 0.1)  # axial spring, no rotation (flexible in y, stiff in rotation)
    flexure(100, 100, 'tx', 'ty', 1, use_symmetry=True)  # axial spring, no shear (flexible in y, stiff in x)
    # flexure(100, 100, 'ty', 'tx', 0.01)  # parallel guiding system (flexible in x, stiff in y)
    # flexure(100, 100, 'rz', 'tx', 0.01)  # parallel guiding system 2 (flexible in x, stiff in rotation)
    # flexure(100, 100, 'ty', 'rz', 0.01, volfrac=0.3)  # notch hinge (flexible in rotation, stiff in y)
    # flexure(100, 50, 'tx', 'rz', 0.01, use_symmetry=True)  # notch hinge 2 (flexible in rotation, stiff in x)
