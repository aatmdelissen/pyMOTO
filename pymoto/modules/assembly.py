"""Assembly modules for finite element analysis"""

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
    r"""Assembles a sparse matrix according to element scaling :math:`\mathbf{A} = \sum_e x_e \mathbf{A}_e`

    Each element matrix is scaled and with the scaling parameter of that element
    :math:`\mathbf{A} = \sum_e x_e \mathbf{A}_e`.
    The number of degrees of freedom per node is deduced from the size of the element matrix passed into the module.
    For instance, in case an element matrix of shape ``(3*4, 3*4)`` gets passed with a 2D :class:`DomainDefinition`, the
    number of dofs per node equals ``3``.

    Input Signal:
        - ``x``: Scaling vector of size ``(Nel)``

    Output Signal:
        - ``A``: System matrix of size ``(n, n)``
    """

    def __init__(
        self,
        domain: DomainDefinition,
        element_matrix: np.ndarray,
        bc=None,
        bcdiagval=None,
        matrix_type=csc_matrix,
        add_constant=None,
    ):
        r"""Initialize assembly module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The domain for which should be assembled
            element_matrix (np.ndarray): The element matrix :math:`\mathbf{K}_e` of size 
              `(#dofs_per_element, #dofs_per_element)`
            bc (optional): Indices of any dofs that are constrained to zero (Dirichlet boundary condition).
              These boundary conditions are enforced by setting the row and column of that dof to zero.
            bcdiagval (optional): Value to put on the diagonal of the matrix at dofs where boundary conditions are 
              active. Default is maximum value of the element matrix.
            matrix_type (optional): The matrix type to construct. This is a constructor which must accept the arguments
              `matrix_type((vals, (row_idx, col_idx)), shape=(n, n))`. Defaults to csc_matrix.
            add_constant (optional): A constant (*e.g.* sparse matrix) to add.
        """
        self.elmat = element_matrix
        self.ndof = self.elmat.shape[-1] // domain.elemnodes  # Number of dofs per node
        self.n = self.ndof * domain.nnodes  # Matrix size
        self.nnode = domain.nnodes

        self.dofconn = domain.get_dofconnectivity(self.ndof)

        # Row and column indices for the matrix
        self.rows = np.kron(self.dofconn, np.ones((1, domain.elemnodes * self.ndof), dtype=int)).flatten()
        self.cols = np.kron(self.dofconn, np.ones((domain.elemnodes * self.ndof, 1), dtype=int)).flatten()
        self.matrix_type = matrix_type

        # Boundary conditions
        self.bcdiagval = np.max(element_matrix) if bcdiagval is None else bcdiagval
        if bc is not None:
            self.bc = np.asarray(bc).flatten()
            bc_inds = np.bitwise_or(np.isin(self.rows, self.bc), np.isin(self.cols, self.bc))
            self.bcselect = np.argwhere(np.bitwise_not(bc_inds)).flatten()
            self.bcrows = np.concatenate((self.rows[self.bcselect], self.bc))
            self.bccols = np.concatenate((self.cols[self.bcselect], self.bc))
        else:
            self.bc = None
            self.bcselect = None
            self.bcrows = self.rows
            self.bccols = self.cols

        self.add_constant = add_constant

    def __call__(self, xscale: np.ndarray):
        nel = self.dofconn.shape[0]
        assert xscale.size == nel, f"Input vector wrong size ({xscale.size}), must be of size #nel ({nel})"
        scaled_el = (self.elmat.flatten() * xscale[..., np.newaxis]).flatten()

        # Set boundary conditions
        if self.bc is not None:
            # Remove entries that correspond to bc before initializing
            mat_values = np.concatenate((scaled_el[self.bcselect], self.bcdiagval * np.ones(len(self.bc))))
        else:
            mat_values = scaled_el

        try:
            mat = self.matrix_type((mat_values, (self.bcrows, self.bccols)), shape=(self.n, self.n))
        except TypeError as e:
            raise type(e)(
                str(e)
                + "\n\tInvalid matrix_type={}. Either scipy.sparse.cscmatrix or "
                "scipy.sparse.csrmatrix are supported".format(self.matrix_type)
            ).with_traceback(sys.exc_info()[2]) from None

        if self.add_constant is not None:
            mat += self.add_constant
        return mat

    def _sensitivity(self, dgdmat: Union[DyadCarrier, np.ndarray]):
        if dgdmat.size <= 0:
            return [None]
        if self.bc is not None:
            dgdmat[self.bc, :] = 0.0
            dgdmat[:, self.bc] = 0.0
        dx = np.zeros_like(self.sig_in[0].state)
        if isinstance(dgdmat, np.ndarray):
            for i in range(len(dx)):
                indu, indv = np.meshgrid(self.dofconn[i], self.dofconn[i], indexing="ij")
                dxi = einsum("ij,ij->", self.elmat, dgdmat[indu, indv])
                dx[i] = np.real(dxi) if np.isrealobj(dx) else dxi
        elif isinstance(dgdmat, DyadCarrier):
            dxi = dgdmat.contract(self.elmat, self.dofconn, self.dofconn)
            dx[:] = np.real(dxi) if np.isrealobj(dx) else dxi
        return dx


def get_B(dN_dx, voigt=True):
    """Gets the strain-displacement relation (Cook, eq 3.1-9, P.80)

      - 1D : [ε_x]_i = B [u]_i
      - 2D : [ε_x; ε_y; γ_xy]_i = B [u, v]_i
      - 3D : [ε_x; ε_y; ε_z; γ_xy; γ_yz; γ_zx]_i = B [u, v, w]_i (standard notation)
             [ε_x; ε_y; ε_z; γ_yz; γ_zx; γ_xy]_i = B [u, v, w]_i (Voigt notation)

    Args:
        dN_dx: Shape function derivatives [dNi_dxj] of size (#shapefn. x #dimensions)
        voigt(optional): Use Voigt notation for the shear terms [yz, zx, xy] or standard notation [xy, yz, zx]

    Returns:
        B strain-displacement relation of size (#strains x #shapefn.*#dimensions)
    """
    n_dim, n_shapefn = dN_dx.shape
    n_strains = int((n_dim * (n_dim + 1)) / 2)  # Triangular number: ndim=3 -> nstrains = 3+2+1
    B = np.zeros((n_strains, n_shapefn * n_dim), dtype=dN_dx.dtype)
    if n_dim == 1:
        for i in range(n_shapefn):
            B[i, 0] = dN_dx[i, 0]
    elif n_dim == 2:
        for i in range(n_shapefn):
            B[:, i * n_dim : (i + 1) * n_dim] = np.array(
                [[dN_dx[0, i], 0], [0, dN_dx[1, i]], [dN_dx[1, i], dN_dx[0, i]]]
            )
    elif n_dim == 3:
        for i in range(n_shapefn):
            if voigt:
                B[:, i * n_dim : (i + 1) * n_dim] = np.array(
                    [
                        [dN_dx[0, i], 0, 0],
                        [0, dN_dx[1, i], 0],
                        [0, 0, dN_dx[2, i]],
                        [0, dN_dx[2, i], dN_dx[1, i]],
                        [dN_dx[2, i], 0, dN_dx[0, i]],
                        [dN_dx[1, i], dN_dx[0, i], 0],
                    ]
                )
            else:
                B[:, i * n_dim : (i + 1) * n_dim] = np.array(
                    [
                        [dN_dx[0, i], 0, 0],
                        [0, dN_dx[1, i], 0],
                        [0, 0, dN_dx[2, i]],
                        [dN_dx[1, i], dN_dx[0, i], 0],
                        [0, dN_dx[2, i], dN_dx[1, i]],
                        [dN_dx[2, i], 0, dN_dx[0, i]],
                    ]
                )
    else:
        raise ValueError(f"Number of dimensions ({n_dim}) cannot be greater than 3")
    return B


def get_D(E: float, nu: float, mode: str = "strain"):
    """Get material constitutive relation for linear elasticity

    Args:
        E: Young's modulus
        nu: Poisson's ratio
        mode: Plane-``strain``, plane-``stress``, or ``3D``

    Returns:
        Material matrix
    """
    mu = E / (2 * (1 + nu))
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    c1 = 2 * mu + lam
    if "strain" in mode.lower():
        return np.array([[c1, lam, 0], [lam, c1, 0], [0, 0, mu]])
    elif "stress" in mode.lower():
        a = E / (1 - nu * nu)
        return a * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    elif "3d" in mode.lower():
        return np.array(
            [
                [c1, lam, lam, 0, 0, 0],
                [lam, c1, lam, 0, 0, 0],
                [lam, lam, c1, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ]
        )
    else:
        raise ValueError("Only for plane-stress, plane-strain, or 3d")


class AssembleStiffness(AssembleGeneral):
    r"""Stiffness matrix assembly by scaling elements in 2D or 3D
    :math:`\mathbf{K} = \sum_e x_e \mathbf{K}_e`

    Input Signal:
        - ``x``: Scaling vector of size ``(Nel)``

    Output Signal:
        - ``K``: Stiffness matrix of size ``(n, n)``
    """

    def __init__(
        self,
        domain: DomainDefinition,
        *args,
        e_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        plane="strain",
        **kwargs,
    ):
        """Initialize stiffness assembly module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The domain to assemble for; this determines the element size 
              and dimensionality
            *args: Other arguments are passed to :py:class:`pymoto.AssembleGeneral`
            e_modulus (float, optional): Young's modulus. Defaults to 1.0.
            poisson_ratio (float, optional): Poisson's ratio. Defaults to 0.3.
            plane (str, optional): Plane `"strain"` or plane `"stress"`. Defaults to `"strain"`.
            **kwargs: Other keyword arguments are passed to :py:class:`pymoto.AssembleGeneral`
        """
        self.E, self.nu = e_modulus, poisson_ratio

        # Get material relation
        D = get_D(self.E, self.nu, "3d" if domain.dim == 3 else plane.lower())
        if domain.dim == 2:
            D *= domain.element_size[2]

        nnode = 2**domain.dim  # Number of nodes per element
        ndof = nnode * domain.dim

        # Element stiffness matrix
        dtype = np.result_type(D, domain.element_size.dtype)
        self.stiffness_element = np.zeros((ndof, ndof), dtype=dtype)

        # Numerical integration
        siz = domain.element_size
        w = np.prod(siz[: domain.dim] / 2)
        for n in domain.node_numbering:
            pos = n * (siz / 2) / np.sqrt(3)  # Sampling point
            dN_dx = domain.eval_shape_fun_der(pos)
            B = get_B(dN_dx)
            self.stiffness_element += w * B.T @ D @ B  # Add contribution

        super().__init__(domain, self.stiffness_element, *args, **kwargs)


class AssembleMass(AssembleGeneral):
    r"""Consistent mass matrix or equivalents assembly by scaling elements
    :math:`\mathbf{M} = \sum_e x_e \mathbf{M}_e`

    Input Signal:
        - ``x``: Scaling vector of size ``(Nel)``

    Output Signal:
        - ``M``: Mass matrix of size ``(n, n)``
    """

    def __init__(
        self,
        domain: DomainDefinition,
        *args,
        material_property: float = 1.0,
        ndof: int = 1,
        bcdiagval: float = 0.0,
        **kwargs,
    ):
        """Initialize mass assembly module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The domain to assemble for; this determines the element size 
              and dimensionality
            *args: Other arguments are passed to :py:class:`pymoto.AssembleGeneral`
            material_property (float, optional): Material property to use in the element matrix (for mass matrix the 
              material density is used; for damping the damping parameter, and for a thermal capacity matrix the thermal
              capacity multiplied with density). Defaults to 1.0.
            ndof (int, optional): Amount of dofs per node (for mass and damping: `ndof = domain.dim`; else `ndof=1`). 
              Defaults to 1.
            bcdiagval (float, optional): The value to put on the diagonal in case of boundary conditions (bc). Defaults 
              to 0.0.
            **kwargs: Other keyword arguments are passed to :py:class:`pymoto.AssembleGeneral`
        """
        # Element mass (or equivalent) matrix
        self.el_mat = np.zeros((domain.elemnodes * ndof, domain.elemnodes * ndof))

        # Numerical integration
        siz = domain.element_size
        w = np.prod(siz[: domain.dim] / 2)
        if domain.dim != 3:
            material_property *= np.prod(siz[domain.dim :])
        Nmat = np.zeros((ndof, domain.elemnodes * ndof))

        for n in domain.node_numbering:
            pos = n * (siz / 2) / np.sqrt(3)  # Sampling point
            N = domain.eval_shape_fun(pos)
            for d in range(domain.elemnodes):
                Nmat[0:ndof, ndof * d : ndof * d + ndof] = (
                    np.identity(ndof) * N[d]
                )  # fill up shape function matrix according to ndof
            self.el_mat += w * material_property * Nmat.T @ Nmat  # Add contribution

        super().__init__(domain, self.el_mat, *args, bcdiagval=bcdiagval, **kwargs)


class AssemblePoisson(AssembleGeneral):
    r"""Assembly of matrix to solve Poisson equation (e.g. Thermal conductivity, Electric permittivity)
    :math:`\mathbf{P} = \sum_e x_e \mathbf{P}_e`

    Input Signal:
        - ``x``: Scaling vector of size ``(Nel)``

    Output Signal:
        - ``P``: Poisson matrix of size ``(n, n)``
    """

    def __init__(self, domain: DomainDefinition, *args, material_property: float = 1.0, **kwargs):
        """Initialize Poisson matrix assembly module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The domain to assemble for; this determines the element size 
              and dimensionality
            *args: Other arguments are passed to :py:class:`pymoto.AssembleGeneral`
            material_property (float, optional): Material property (*e.g.* thermal conductivity, electric permittivity). 
              Defaults to 1.0.
            **kwargs: Other keyword arguments are passed to :py:class:`pymoto.AssembleGeneral`
        """
        # Prepare material properties and element matrices
        self.material_property = material_property
        self.poisson_element = np.zeros((domain.elemnodes, domain.elemnodes))

        # Numerical Integration
        siz = domain.element_size
        w = np.prod(siz[: domain.dim] / 2)
        if domain.dim != 3:
            self.material_property *= siz[domain.dim :]

        for n in domain.node_numbering:
            pos = n * (siz / 2) / np.sqrt(3)  # Sampling point
            Bn = domain.eval_shape_fun_der(pos)
            self.poisson_element += w * self.material_property * Bn.T @ Bn  # Add contribution

        super().__init__(domain, self.poisson_element, *args, **kwargs)


class ElementOperation(Module):
    r"""Generic module for element-wise operations based on nodal information

    :math:`y_e = \mathbf{B} \mathbf{u}_e`

    This module is the reverse of :py:class:`pymoto.NodalOperation`.

    Input Signal:
        - ``u``: Nodal vector of size ``(#dofs_per_node * #nodes)``

    Output Signal:
        - ``y``: Elemental output data of size ``(..., #elements)`` or ``(#dofs, ..., #elements)``
    """

    def __init__(self, domain: DomainDefinition, element_matrix: np.ndarray):
        r"""Initialize element operation module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The finite element domain
            element_matrix (np.ndarray): The element operator matrix :math:`\mathbf{B}` of size
              ``(..., #dofs_per_element)`` or ``(..., #nodes_per_element)``
        """
        if element_matrix.shape[-1] % domain.elemnodes != 0:
            raise IndexError(
                f"Size of last dimension of element operator matrix ({element_matrix.shape[-1]}) is not compatible "
                f"with mesh. Must be dividable by the number of nodes per element ({domain.elemnodes})."
            )
        self.domain = domain
        self.element_matrix = element_matrix
        self.dofconn = None

    def __call__(self, u):
        if u.size % self.domain.nnodes != 0:
            raise IndexError(f"Size of input vector ({u.size}) does not match number of nodes ({self.domain.nnodes})")
        ndof = u.size // self.domain.nnodes

        if self.element_matrix.shape[-1] != self.domain.elemnodes * ndof:
            # Initialize only after first call to response(), because the number of dofs may not yet be known
            em = self.element_matrix.copy()
            msg = (
                f"Size of element matrix must match #dofs_per_element ({ndof * self.domain.elemnodes})",
                f" or #nodes_per_element ({self.domain.elemnodes}).",
            )
            assert em.shape[-1] == self.domain.elemnodes, msg

            # Element matrix is repeated for each dof
            self.element_matrix = np.zeros((ndof, *self.element_matrix.shape[:-1], ndof * self.domain.elemnodes))
            for i in range(ndof):
                self.element_matrix[i, ..., i::ndof] = em

        if self.dofconn is None:
            self.dofconn = self.domain.get_dofconnectivity(ndof)

        assert self.element_matrix.shape[-1] == ndof * self.domain.elemnodes
        return einsum("...k, lk -> ...l", self.element_matrix, u[self.dofconn], optimize=True)

    def _sensitivity(self, dy):
        du_el = einsum("...k, ...l -> lk", self.element_matrix, dy, optimize=True)
        du = np.zeros_like(self.sig_in[0].state)
        np.add.at(du, self.dofconn, du_el)
        return du


class Strain(ElementOperation):
    r"""Evaluate average mechanical strains in solid elements based on deformation

    The strains are returned in Voigt notation.
    :math:`\mathbf{\epsilon}_e = \mathbf{B} \mathbf{u}_e`

    Each integration point in the element has different strain values. Here, the average is returned.

    The returned strain is either
    :math:`\mathbf{\epsilon} = \begin{bmatrix}\epsilon_{xx} & \epsilon_{yy} & \epsilon_{xy} \end{bmatrix}`
    in case ``voigt = False`` or
    :math:`\mathbf{\epsilon} = \begin{bmatrix}\epsilon_{xx} & \epsilon_{yy} & \gamma_{xy} \end{bmatrix}`
    in case ``voigt = True``, for which :math:`\gamma_{xy}=2\epsilon_{xy}`.

    Input Signal:
       - ``u``: Nodal vector of size ``(#dofs_per_node * #nodes)``

    Output Signal:
       - ``e``: Strain matrix of size ``(#strains_per_element, #elements)``
    """

    def __init__(self, domain: DomainDefinition, voigt: bool = True):
        """Initialize strain evaluation module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The finite element domain
            voigt (bool, optional): Use Voigt strain notation (2x off-diagonal strain contribution). Defaults to True.
        """
        # Numerical integration
        B = None
        siz = domain.element_size
        w = 1 / domain.elemnodes  # Average strain at the integration points
        for n in domain.node_numbering:
            pos = n * (siz / 2) / np.sqrt(3)  # Sampling point
            dN_dx = domain.eval_shape_fun_der(pos)
            B_add = w * get_B(dN_dx)  # Add contribution
            if B is None:
                B = B_add
            else:
                B += B_add

        if voigt:
            idx_shear = np.count_nonzero(B, axis=1) == 2 * domain.elemnodes  # Shear is combination of two displacements
            B[idx_shear, :] *= 2  # Use engineering strain

        super().__init__(domain, B)


class Stress(Strain):
    """Calculate the average stresses per element

    Input Signal:
       - ``u``: Nodal vector of size ``(#dofs_per_node * #nodes)``

    Output Signal:
       - ``s``: Stress matrix of size ``(#stresses_per_element, #elements)``
    """

    def __init__(
        self, domain: DomainDefinition, e_modulus: float = 1.0, poisson_ratio: float = 0.3, plane: str = "strain"
    ):
        """Initialize stress evaluation module

        Args:
            Use Voigt strain notation (2x off-diagonal strain contribution)
            e_modulus (float, optional): Young's modulus. Defaults to 1.0.
            poisson_ratio (float, optional): Poisson ratio. Defaults to 0.3.
            plane (str, optional): Plane `"strain"` or `"stress"`. Defaults to `"strain"`.
        """
        super().__init__(domain, voigt=True)

        # Get material relation
        D = get_D(e_modulus, poisson_ratio, "3d" if domain.dim == 3 else plane.lower())
        if domain.dim == 2:
            D *= domain.element_size[2]
        self.element_matrix = D @ self.element_matrix


class ElementAverage(ElementOperation):
    r"""Determine average value in element of input nodal values

    Input Signal:
       - ``v``: Nodal vector of size ``(#dofs_per_node * #nodes)``

    Output Signal:
       - ``v_el``: Elemental vector of size ``(#elements)`` or ``(#dofs, #elements)`` if ``#dofs_per_node>1``
    """

    def __init__(self, domain: DomainDefinition):
        """Initialize element average module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The finite element domain
        """
        shapefuns = domain.eval_shape_fun(pos=np.array([0, 0, 0]))
        super().__init__(domain, shapefuns)


class NodalOperation(Module):
    r"""Generic module for nodal operations based on elemental information

    :math:`u_e = \mathbf{A} x_e`

    This module is the reverse of :py:class:`pymoto.ElementOperation`.

    Input Signal:
        - ``x``: Elemental vector of size ``(#elements)``

    Output Signal:
        - ``u``: nodal output data of size ``(..., #dofs_per_node * #nodes)``
    """

    def __init__(self, domain: DomainDefinition, element_matrix: np.ndarray):
        r"""Initialize nodal operation module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The finite element domain
            element_matrix (np.ndarray): The element operator matrix :math:`\mathbf{A}` of size 
              ``(..., #dofs_per_element)``
        """
        if element_matrix.shape[-1] % domain.elemnodes != 0:
            raise IndexError(
                "Size of last dimension of element operator matrix is not compatible with mesh. "
                "Must be dividable by the number of nodes."
            )

        ndof = element_matrix.shape[-1] // domain.elemnodes

        self.element_matrix = element_matrix
        self.dofconn = domain.get_dofconnectivity(ndof)
        self.ndofs = ndof * domain.nnodes

    def __call__(self, x):
        dofs_el = einsum("...k, ...l -> lk", self.element_matrix, x, optimize=True)
        dofs = np.zeros(self.ndofs)
        np.add.at(dofs, self.dofconn, dofs_el)
        return dofs

    def _sensitivity(self, dx):
        return einsum("...k, lk -> ...l", self.element_matrix, dx[self.dofconn], optimize=True)


class ThermoMechanical(NodalOperation):
    r"""Determine equivalent thermo-mechanical load from design vector and elemental temperature difference

    :math:`f_thermal = \mathbf{A} (x*t_delta)_e`

    Input Signal:
        - ``x*t_delta``: Elemental vector of size ``(#elements)`` containing elemental densities multiplied by
                         elemental temperature difference

    Output Signal:
        - ``f_thermal``: nodal equivalent thermo-mechanical load of size ``(#dofs_per_node * #nodes)``
    """

    def __init__(
        self,
        domain: DomainDefinition,
        e_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        alpha: float = 1e-6,
        plane: str = "strain",
    ):
        """Initalize thermo-mechanical load module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The finite element domain
            e_modulus (float, optional): Young's modulus. Defaults to 1.0.
            poisson_ratio (float, optional): Poisson ratio. Defaults to 0.3.
            alpha (float, optional): Coefficient of thermal expansion. Defaults to 1e-6.
            plane (str, optional): plane (str, optional): Plane `"strain"` or `"stress"`. Defaults to `"strain"`.
        """
        dim = domain.dim
        D = get_D(e_modulus, poisson_ratio, "3d" if dim == 3 else plane.lower())
        if dim == 2:
            Phi = np.array([1, 1, 0])
            D *= domain.element_size[2]
        elif dim == 3:
            Phi = np.array([1, 1, 1, 0, 0, 0])

        # Numerical integration
        BDPhi = np.zeros(domain.elemnodes * dim)
        siz = domain.element_size
        w = np.prod(siz[: domain.dim] / 2)
        for n in domain.node_numbering:
            pos = n * (siz / 2) / np.sqrt(3)  # Sampling point
            dN_dx = domain.eval_shape_fun_der(pos)
            B = get_B(dN_dx)
            BDPhi += w * B.T @ D @ Phi  # Add contribution

        super().__init__(domain, alpha * BDPhi)
