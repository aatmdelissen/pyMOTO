""" Assembly modules for finite element analysis """
import sys
from typing import Union

import numpy as np
from scipy.sparse import csc_matrix

from pymoto import Module, DyadCarrier, DomainDefinition

try:
    from opt_einsum import contract as einsum
except ModuleNotFoundError:
    from numpy import einsum


class AssembleGeneral(Module):
    r""" Assembles a sparse matrix according to element scaling :math:`\mathbf{A} = \sum_e x_e \mathbf{A}_e`

    Each element matrix is scaled and with the scaling parameter of that element
    :math:`\mathbf{A} = \sum_e x_e \mathbf{A}_e`.
    The number of degrees of freedom per node is deduced from the size of the element matrix passed into the module.
    For instance, in case an element matrix of shape ``(3*4, 3*4)`` gets passed with a 2D :class:`DomainDefinition`, the
    number of dofs per node equals ``3``.

    Input Signal:
        - ``x``: Scaling vector of size ``(Nel)``

    Output Signal:
        - ``A``: system matrix of size ``(n, n)``

    Args:
        domain: The domain-definition for which should be assembled
        element_matrix: The element matrix for one element :math:`\mathbf{K}_e`
        bc (optional): Indices of any dofs that are constrained to zero (Dirichlet boundary condition).
          These boundary conditions are enforced by setting the row and column of that dof to zero.
        bcdiagval (optional): Value to put on the diagonal of the matrix at dofs where boundary conditions are active.
        matrix_type (optional): The matrix type to construct. This is a constructor which must accept the arguments
          ``matrix_type((vals, (row_idx, col_idx)), shape=(n, n))``
        add_constant (optional): A constant (e.g. matrix) to add.
    """

    def _prepare(self, domain: DomainDefinition, element_matrix: np.ndarray, bc=None, bcdiagval=None,
                 matrix_type=csc_matrix, add_constant=None):
        self.elmat = element_matrix
        self.ndof = self.elmat.shape[-1] // domain.elemnodes  # Number of dofs per node
        self.n = self.ndof * domain.nnodes  # Matrix size
        self.nnode = domain.nnodes

        self.dofconn = domain.get_dofconnectivity(self.ndof)

        # Row and column indices for the matrix
        self.rows = np.kron(self.dofconn, np.ones((domain.elemnodes*self.ndof, 1), dtype=int)).flatten()
        self.cols = np.kron(self.dofconn, np.ones((1, domain.elemnodes*self.ndof), dtype=int)).flatten()
        self.matrix_type = matrix_type

        # Boundary conditions
        self.bc = bc
        self.bcdiagval = np.max(element_matrix) if bcdiagval is None else bcdiagval
        if bc is not None:
            self.bcselect = np.argwhere(np.bitwise_not(np.bitwise_or(np.isin(self.rows, self.bc),
                                                                     np.isin(self.cols, self.bc)))).flatten()

            self.rows = np.concatenate((self.rows[self.bcselect], self.bc))
            self.cols = np.concatenate((self.cols[self.bcselect], self.bc))
        else:
            self.bcselect = None

        self.add_constant = add_constant

    def _response(self, xscale: np.ndarray):
        scaled_el = ((self.elmat.flatten()[np.newaxis]).T * xscale).flatten(order='F')

        # Set boundary conditions
        if self.bc is not None:
            # Remove entries that correspond to bc before initializing
            mat_values = np.concatenate((scaled_el[self.bcselect], self.bcdiagval*np.ones(len(self.bc))))
        else:
            mat_values = scaled_el

        try:
            mat = self.matrix_type((mat_values, (self.rows, self.cols)), shape=(self.n, self.n))
        except TypeError as e:
            raise type(e)(str(e) + "\n\tInvalid matrix_type={}. Either scipy.sparse.cscmatrix or "
                                   "scipy.sparse.csrmatrix are supported"
                          .format(self.matrix_type)).with_traceback(sys.exc_info()[2]) from None

        if self.add_constant is not None:
            mat += self.add_constant
        return mat

    def _sensitivity(self, dgdmat: Union[DyadCarrier, np.ndarray]):
        if dgdmat.size <= 0:
            return [None]
        if self.bc is not None:
            dgdmat[self.bc, :] = 0.0
            dgdmat[:, self.bc] = 0.0
        if isinstance(dgdmat, np.ndarray):
            dx = np.zeros_like(self.sig_in[0].state)
            for i in range(len(dx)):
                indu, indv = np.meshgrid(self.dofconn[i], self.dofconn[i], indexing='ij')
                dx[i] = einsum("ij,ij->", self.elmat, dgdmat[indu, indv])
            return dx
        elif isinstance(dgdmat, DyadCarrier):
            return dgdmat.contract(self.elmat, self.dofconn, self.dofconn)


def get_B(dN_dx):
    """ Gets the strain-displacement relation (Cook, eq 3.1-9, P.80)

      - 1D : [ε_x]_i = B [u]_i
      - 2D : [ε_x; ε_y; γ_xy]_i = B [u, v]_i
      - 3D : [ε_x; ε_y; ε_z; γ_xy; γ_yz; γ_zx]_i = B [u, v, w]_i

    Args:
        dN_dx: Shape function derivatives [dNi_dxj] of size (#shapefn. x #dimensions)

    Returns:
        B strain-displacement relation of size (#strains x #shapefn.*#dimensions)
    """
    n_dim, n_shapefn = dN_dx.shape
    n_strains = int((n_dim * (n_dim+1))/2)  # Triangular number: ndim=3 -> nstrains = 3+2+1
    B = np.zeros((n_strains, n_shapefn*n_dim))
    if n_dim == 1:
        for i in range(n_shapefn):
            B[i, 0] = dN_dx[i, 0]
    elif n_dim == 2:
        for i in range(n_shapefn):
            B[:, i*n_dim:(i+1)*n_dim] = np.array([[dN_dx[0, i], 0],
                                                  [0,           dN_dx[1, i]],
                                                  [dN_dx[1, i], dN_dx[0, i]]])
    elif n_dim == 3:
        for i in range(n_shapefn):
            B[:, i*n_dim:(i+1)*n_dim] = np.array([[dN_dx[0, i], 0,           0],
                                                  [0,           dN_dx[1, i], 0],
                                                  [0,           0,           dN_dx[2, i]],
                                                  [dN_dx[1, i], dN_dx[0, i], 0],
                                                  [0,           dN_dx[2, i], dN_dx[1, i]],
                                                  [dN_dx[2, i], 0,           dN_dx[0, i]]])
    else:
        raise ValueError(f"Number of dimensions ({n_dim}) cannot be greater than 3")
    return B


def get_D(E: float, nu: float, mode: str = 'strain'):
    """ Get material constitutive relation for linear elasticity

    Args:
        E: Young's modulus
        nu: Poisson's ratio
        mode: Plane-``strain``, plane-``stress``, or ``3D``

    Returns:
        Material matrix
    """
    mu = E/(2*(1+nu))
    lam = (E*nu) / ((1+nu)*(1-2*nu))
    c1 = 2*mu + lam
    if 'strain' in mode.lower():
        return np.array([[c1,  lam, 0],
                         [lam, c1,  0],
                         [0,   0,   mu]])
    elif 'stress' in mode.lower():
        a = E / (1 - nu * nu)
        return a*np.array([[1,  nu, 0],
                           [nu, 1,  0],
                           [0,  0,  (1-nu)/2]])
    elif '3d' in mode.lower():
        return np.array([[c1,  lam, lam, 0,  0,  0],
                         [lam, c1,  lam, 0,  0,  0],
                         [lam, lam, c1,  0,  0,  0],
                         [0,   0,   0,   mu, 0,  0],
                         [0,   0,   0,   0,  mu, 0],
                         [0,   0,   0,   0,  0,  mu]])
    else:
        raise ValueError("Only for plane-stress, plane-strain, or 3d")


class AssembleStiffness(AssembleGeneral):
    r""" Stiffness matrix assembly by scaling elements in 2D or 3D
    :math:`\mathbf{K} = \sum_e x_e \mathbf{K}_e`

    Input Signal:
        - ``x``: Scaling vector of size ``(Nel)``

    Output Signal:
        - ``K``: Stiffness matrix of size ``(n, n)``

    Args:
        domain: The domain to assemble for -- this determines the element size and dimensionality
        args (optional): Other arguments are passed to AssembleGeneral

    Keyword Args:
        e_modulus: Young's modulus
        poisson_ratio: Poisson's ratio
        plane: Plane ``strain`` or plane ``stress``
        bcdiagval: The value to put on the diagonal in case of boundary conditions (bc)
        kwargs: Other keyword-arguments are passed to AssembleGeneral
    """
    def _prepare(self, domain: DomainDefinition, *args,
                 e_modulus: float = 1.0, poisson_ratio: float = 0.3, plane='strain', **kwargs):
        self.E, self.nu = e_modulus, poisson_ratio

        # Get material relation
        D = get_D(self.E, self.nu, '3d' if domain.dim == 3 else plane.lower())
        if domain.dim == 2:
            D *= domain.element_size[2]

        nnode = 2**domain.dim  # Number of nodes per element
        ndof = nnode*domain.dim

        # Element stiffness matrix
        self.stiffness_element = np.zeros((ndof, ndof))

        # Numerical integration
        siz = domain.element_size
        w = np.prod(siz[:domain.dim]/2)
        for n in domain.node_numbering:
            pos = n*(siz/2)/np.sqrt(3)  # Sampling point
            dN_dx = domain.eval_shape_fun_der(pos)
            B = get_B(dN_dx)
            self.stiffness_element += w * B.T @ D @ B  # Add contribution

        super()._prepare(domain, self.stiffness_element, *args, **kwargs)


class AssembleMass(AssembleGeneral):
    r""" Consistent mass matrix or equivalents assembly by scaling elements
    :math:`\mathbf{M} = \sum_e x_e \mathbf{M}_e`

    Input Signal:
        - ``x``: Scaling vector of size ``(Nel)``

    Output Signal:
        - ``M``: Mass matrix of size ``(n, n)``

    Args:
        domain: The domain to assemble for -- this determines the element size and dimensionality
        ndof: Amount of dofs per node (for mass and damping: ndof = domain.dim; else ndof=1)
        *args: Other arguments are passed to AssembleGeneral

    Keyword Args:
        material_property: Material property to use in the element matrix (for mass matrix the material density is used;
            for damping the damping parameter, and for a thermal capacity matrix the thermal capacity multiplied with
            density)
        bcdiagval: The value to put on the diagonal in case of boundary conditions (bc)
        **kwargs: Other keyword-arguments are passed to AssembleGeneral
    """

    def _prepare(self, domain: DomainDefinition, *args, material_property: float = 1.0, ndof: int = 1,
                 bcdiagval: float = 0.0, **kwargs):
        # Element mass (or equivalent) matrix
        self.el_mat = np.zeros((domain.elemnodes * ndof, domain.elemnodes * ndof))

        # Numerical integration
        siz = domain.element_size
        w = np.prod(siz[:domain.dim] / 2)
        if domain.dim != 3:
            material_property *= np.prod(siz[domain.dim:])
        Nmat = np.zeros((ndof, domain.elemnodes * ndof))

        for n in domain.node_numbering:
            pos = n * (siz / 2) / np.sqrt(3)  # Sampling point
            N = domain.eval_shape_fun(pos)
            for d in range(domain.elemnodes):
                Nmat[0:ndof, ndof * d:ndof * d + ndof] = np.identity(ndof) * N[d]  # fill up shape function matrix according to ndof
            self.el_mat += w * material_property * Nmat.T @ Nmat  # Add contribution

        super()._prepare(domain, self.el_mat, *args, bcdiagval=bcdiagval, **kwargs)


class AssemblePoisson(AssembleGeneral):
    r""" Assembly of matrix to solve Poisson equation (e.g. Thermal conductivity, Electric permittivity)
    :math:`\mathbf{P} = \sum_e x_e \mathbf{P}_e`

    Input Signal:
        - ``x``: Scaling vector of size ``(Nel)``

    Output Signal:
        - ``P``: Poisson matrix of size ``(n, n)``

    Args:
        domain: The domain to assemble for -- this determines the element size and dimensionality
        args (optional): Other arguments are passed to AssembleGeneral

    Keyword Args:
        material_property: Material property (e.g. thermal conductivity, electric permittivity)
        bcdiagval: The value to put on the diagonal in case of boundary conditions (bc)
        kwargs: Other keyword-arguments are passed to AssembleGeneral
    """

    def _prepare(self, domain: DomainDefinition, *args, material_property: float = 1.0, **kwargs):
        # Prepare material properties and element matrices
        self.material_property = material_property
        self.poisson_element = np.zeros((domain.elemnodes, domain.elemnodes))

        # Numerical Integration
        siz = domain.element_size
        w = np.prod(siz[:domain.dim]/2)
        if domain.dim != 3:
            self.material_property *= siz[domain.dim:]

        for n in domain.node_numbering:
            pos = n*(siz/2)/np.sqrt(3)  # Sampling point
            Bn = domain.eval_shape_fun_der(pos)
            self.poisson_element += w * self.material_property * Bn.T @ Bn  # Add contribution

        super()._prepare(domain, self.poisson_element, *args, **kwargs)
