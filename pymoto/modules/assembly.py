"""Assembly modules for finite element analysis"""

import sys
from typing import Union, Iterable
import warnings

import numpy as np
import scipy.sparse as sps
from pymoto import Module, DyadCarrier, DomainDefinition
from pymoto.utils import _parse_to_list

try:
    from opt_einsum import contract as einsum
except ModuleNotFoundError:
    from numpy import einsum


class AssembleGeneral(Module):
    r"""Assembles a sparse matrix according to element scaling :math:`\mathbf{A} = \sum_e x_e \mathbf{A}_e`

    Each element matrix is scaled and with the scaling parameter of that element
    :math:`\mathbf{A} = \sum_e \sum_i x_{i,e} \mathbf{A}_{i,e}`.
    The number of degrees of freedom per node is deduced from the size of the element matrix passed into the module.
    For instance, in case an element matrix of shape ``(3*4, 2*4)`` gets passed with a 2D :class:`DomainDefinition`, the
    number of dofs per node equals ``3`` in the row-direction and ``2`` in the column direction.

    Non-square matrices and complex values are supported. Dirichlet boundary conditions are possible to set, which are
    implemented by setting the entire row and column of the constrained dof to zero, and placing a finite value on the 
    diagonal (optionally).

    Input Signal:
        - ``*x``: Scaling vector(s) of size ``(Nel)``

    Output Signal:
        - ``A``: System matrix of size ``(m, n)``
    """

    def __init__(
        self,
        domain: DomainDefinition,
        element_matrix: Union[np.ndarray, Iterable[np.ndarray]],
        bc=None,
        bcdiagval=None,
        matrix_type: type = sps.csr_matrix,
        add_constant=None,
        reuse_sparsity: bool = True,
    ):
        r"""Initialize assembly module

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The domain for which should be assembled
            element_matrix (np.ndarray or List of np.ndarray): The element matrix :math:`\mathbf{K}_e` of size
              `(#dofs_per_element, #dofs_per_element)`. Multiple element matrices can also be provied
            bc (optional): Indices of any dofs that are constrained to zero (Dirichlet boundary condition).
              These boundary conditions are enforced by setting the row and column of that dof to zero.
            bcdiagval (optional): Value to put on the diagonal of the matrix at dofs where boundary conditions are
              active. Default is maximum value of the element matrix.
            matrix_type (optional): The matrix type to construct. This is a constructor which must accept the arguments
              `matrix_type((vals, (row_idx, col_idx)), shape=(n, n))`. Defaults to `scipy.sparse.csc_matrix`.
            add_constant (optional): A constant (*e.g.* sparse matrix) to add. This is added after setting Dirichlet 
              boundary conditions.
            reuse_sparsity (bool, optional): Reuse the sparsity pattern of the sparse matrix. This improves performance 
              during iterations at the cost of a longer setup time.
        """
        # Parse element matrices and obtain shape of the system matrix
        self.elmat = _parse_to_list(element_matrix)
        self.nmat = len(self.elmat)
        if self.nmat < 1:
            raise ValueError("No or invalid element-matrix is given")
        if self.elmat[0].shape[0] % domain.elemnodes != 0:
            raise ValueError("Number of rows in element matrix should be a multiple of the number of nodes per element")
        if self.elmat[0].shape[1] % domain.elemnodes != 0:
            raise ValueError("Number of cols in element matrix should be a multiple of the number of nodes per element")
        self.mdof = self.elmat[0].shape[0] // domain.elemnodes  # Number of dofs per node
        self.ndof = self.elmat[0].shape[1] // domain.elemnodes
        for i in range(1, self.nmat):
            if self.elmat[i].shape != self.elmat[0].shape:
                raise ValueError(
                    f"Element matrices must be the same shape {self.elmat[i].shape} != {self.elmat[0].shape}")
        # Matrix size
        is_square = self.mdof == self.ndof
        self.m = self.mdof * domain.nnodes  # Rows
        self.n = self.ndof * domain.nnodes  # Cols
        self.nel = domain.nel

        # Determine BC diagonal value
        self.bc = None
        self.bcdiagval = bcdiagval
        if bc is not None:
            if not is_square:
                raise NotImplementedError("Passing Dirichlet boundary conditions only works for square matrices")
            self.bc = np.asarray(bc).ravel()

            if bcdiagval is None:
                esum = self.elmat[0]
                for i in range(1, self.nmat):
                    esum = esum + self.elmat[i]
                self.bcdiagval = np.max(esum)

        # Get matrix sparsity pattern
        self.dofconn_row = domain.get_dofconnectivity(self.mdof)  # Element connectivity
        self.dofconn_col = self.dofconn_row if is_square else domain.get_dofconnectivity(self.ndof)
            
        intT = int  # Index integer type

        self.matrix_type = matrix_type  # Set matrix type
        if reuse_sparsity and self.matrix_type not in (sps.csc_matrix, sps.csr_matrix):
            warnings.warn("Reusing sparsity pattern for matrix type {self.matrix_type} not implemented")
        self.reuse_sparsity = reuse_sparsity and self.matrix_type in (sps.csc_matrix, sps.csr_matrix)

        if self.reuse_sparsity:
            # indices: Column index for each value (CSR) or Row index (CSC)
            # indptr: Data index for each row (CSR) or column (CSC)
            sparsity_method = 'manual'
            if sparsity_method == 'unique':
                rows = np.kron(self.dofconn_row, np.ones((1, domain.elemnodes * self.ndof), dtype=intT)).ravel()
                cols = np.kron(self.dofconn_col, np.ones((domain.elemnodes * self.mdof, 1), dtype=intT)).ravel()

                # -------- METHOD BASED ON UNIQUE -------
                # This way of constructing csr/csc data structure is much easier, but also much slower...
                if self.matrix_type == sps.csc_matrix:
                    rc = np.vstack((cols, rows))
                else:  # CSR
                    rc = np.vstack((rows, cols))
                unique_rc, unique_idx, self.datamap = np.unique(rc, axis=1, return_index=True, return_inverse=True)

                self.indices = unique_rc[1, :]
                maxval = self.n if self.matrix_type == sps.csc_matrix else self.m
                self.indptr = np.argwhere(np.diff(unique_rc[0, :], prepend=-1, append=maxval)).ravel()
            elif sparsity_method == 'manual':
                # -------- MANUAL CONSTRUCITON OF CSR/CSC STRUCTURE -------
                # self.indices = np.zeros(nnz)
                # self.indptr = np.zeros(self.m + 1)
                if self.matrix_type == sps.csr_matrix:
                    row_conn, col_conn = self.dofconn_row, self.dofconn_col
                    rows_per_el, cols_per_el = self.mdof * domain.elemnodes, self.ndof * domain.elemnodes
                    num_rows = self.m
                else:  # Reverse row and col for CSC
                    row_conn, col_conn = self.dofconn_col, self.dofconn_row,
                    rows_per_el, cols_per_el = self.ndof * domain.elemnodes, self.mdof * domain.elemnodes
                    num_rows = self.n

                # --- Sort row indices of all elements
                row_indices_flat = row_conn.ravel()  # List of all rows                           (#rows/el * #el, )
                index = np.argsort(row_indices_flat)  # Sort the rows from low to high            (#rows/el * #el, )
                rows_sorted = row_indices_flat[index]  # Row index for each node                  (#rows/el * #el, )
                from_el = index // rows_per_el  # Element corresponding to each the row           (#rows/el * #el, )
                irow, counts = np.unique(rows_sorted, return_counts=True)  # Get unique row numbers (#rows, )
                cmax = counts.max()  # Maximum number of elements per row
                max_cols_per_row = cmax * cols_per_el  # Maximum number of columns per row

                # --- Sort column indices of all elements
                # List of columns in each row: Not all rows have the same number of columns, so -1 denotes there is no 
                # column in that row
                column_map = -np.ones((num_rows, max_cols_per_row), dtype=intT)  # (#rows, #maxcols/row)
                # Select vector for first entry of each unique value
                select = np.argwhere(np.diff(rows_sorted, prepend=-1)>0).ravel()  
                for i in range(cmax):
                    # Set correct columns in each row
                    column_map[irow, i*cols_per_el:(i+1)*cols_per_el] =  col_conn[from_el[select]]
                    # Increment selector and look for the next value. If no values are remaining, remove from the set
                    irow = irow[counts>1]
                    select = select[counts>1] + 1
                    counts = counts[counts>1] - 1

                # Sort column indices per row
                idxsort_colmap = np.argsort(column_map, axis=1)
                sorted_column_map = column_map[np.arange(num_rows)[:, None], idxsort_colmap]
                valid_col_entries = sorted_column_map >= 0  # All valid column entries
                # Select unique entries in the column map
                diff_sorted_cols = np.diff(sorted_column_map, axis=1, prepend=-1)
                select_unique = np.logical_and(diff_sorted_cols > 0, valid_col_entries)
                
                # --- Construct CSR data layout
                # Select all valid columns for the column indices (sorted row first, then column)
                self.indices = sorted_column_map[select_unique]  # (nnz, )
                nnz = self.indices.size

                # Get row index pointer (number of columns per row)
                self.indptr = np.zeros(num_rows + 1, dtype=intT)
                np.cumsum(np.sum(select_unique, axis=1), out=self.indptr[1:])

                # Data index (reverse mapping)
                target_data_indices = np.arange(nnz)  # Indices of the sparse matrix data-vector (nnz, )
                sorted_target_data = -np.ones_like(sorted_column_map)  
                sorted_target_data[select_unique] = target_data_indices
                
                for i in range(cmax):
                    # Fill next duplicate entry with the same data-target index
                    sel = np.logical_and(valid_col_entries, sorted_target_data < 0)
                    tar = np.roll(sel, -1, axis=1)  
                    sorted_target_data[sel] = sorted_target_data[tar]

                # Reverse sort columns
                unsorted_target_data = np.empty_like(sorted_column_map)
                unsorted_target_data[np.arange(num_rows)[:, None], idxsort_colmap] = sorted_target_data

                # Reverse sort rows
                orig_data = np.zeros((domain.nel * rows_per_el, cols_per_el), dtype=intT)
                orig_data[index[None, :], :] = unsorted_target_data[unsorted_target_data>=0].reshape((-1, cols_per_el))
                
                if self.matrix_type == sps.csc_matrix:
                    # Extra transpose for the element matrix
                    self.datamap = np.swapaxes(orig_data.reshape((-1, rows_per_el, cols_per_el)), 1, 2).ravel()
                else:
                    self.datamap = orig_data.ravel()
            else:
                raise NotImplementedError(f"Sparsity method {sparsity_method} not implemented")
        
            # Boundary conditions
            if self.bc is not None:
                max_rc = self.n if self.matrix_type == sps.csc_matrix else self.m
                row_indices = np.repeat(np.arange(max_rc), self.indptr[1:] - self.indptr[:-1])
                bc_inds = np.logical_or(np.isin(self.indices, self.bc), np.isin(row_indices, self.bc))[self.datamap]
 
                # Find data indices for diagonal bc entries
                idx0 = self.indptr[self.bc]  # Select columns (or rows)
                idx1 = self.indptr[self.bc+1]
                max_delta = (idx1 - idx0).max()
                idx_range = idx0[:, None] + np.arange(max_delta)  # range of indices (bc indices, data indices)
                valid = idx_range < idx1[:, None]  # mask for valid range of indices

                rows = -1 * np.ones_like(idx_range) 
                rows[valid] = self.indices[idx_range[valid]]  # Get which rows are corresponding (or cols)
                
                self.bcadd = idx_range[rows == self.bc[:, None]]

                # Adapt datamap such that we don't need to slice the input data: the values are just added to the 
                # diagonal, which is then overwritten with the diagonal value
                self.datamap[bc_inds] = self.bcadd[0]
        else:
            rows = np.kron(self.dofconn_row, np.ones((1, domain.elemnodes * self.ndof), dtype=intT)).ravel()
            cols = np.kron(self.dofconn_col, np.ones((domain.elemnodes * self.mdof, 1), dtype=intT)).ravel()
            # Boundary conditions
            if self.bc is not None:
                bc_inds = np.bitwise_or(np.isin(rows, self.bc), np.isin(cols, self.bc))
                self.bcselect = np.argwhere(np.bitwise_not(bc_inds)).ravel()
                self.rows = rows[self.bcselect]
                self.cols = cols[self.bcselect]
            else:
                self.rows, self.cols = rows, cols
       
        self.add_constant = add_constant

    def __call__(self, *xscale: np.ndarray):
        if len(xscale) != self.nmat:
            raise ValueError(f"One scaling vector must be given for each element matrix ({self.nmat})")
    
        dtype = np.result_type(*xscale, *self.elmat)  # Determine dtype
        for x in xscale:
            if x.size != self.nel:
                raise ValueError(f"Input vector wrong size ({x.size}), must be equal to #nel ({self.nel})")

        # Calculate scaled element data
        scaled_el = None
        for x, m in zip(xscale, self.elmat):
            el_add = (m.ravel()[None, :] * x.ravel()[:, None]).ravel()
            if scaled_el is None:
                scaled_el = el_add.astype(dtype, copy=False)
            else:
                scaled_el += el_add

        if self.reuse_sparsity:
            # NOTE: The actual matrix cannot be re-used, because scipy sometimes does some pruning on the background
            
            # Calculate data
            data = np.zeros_like(self.indices, dtype=dtype)
            np.add.at(data, self.datamap, scaled_el)
            
            if self.bc is not None:
                # Set boundary condition diagonal values
                data[self.bcadd] = self.bcdiagval

            # Construct sparse matrix
            mat = self.matrix_type((data, self.indices, self.indptr), shape=(self.m, self.n))
        else:
            vals = scaled_el[self.bcselect] if self.bc is not None else scaled_el

            try:
                mat = self.matrix_type((vals, (self.rows, self.cols)), shape=(self.m, self.n))
            except TypeError as e:
                raise type(e)(
                    str(e)
                    + f"Invalid matrix_type={self.matrix_type}. Either scipy.sparse.cscmatrix or "
                    "scipy.sparse.csrmatrix are supported".format(self.matrix_type)
                ).with_traceback(sys.exc_info()[2]) from None
            
            if self.bc is not None:
                # Add diagonal entry
                diag = np.zeros(min(mat.shape))
                diag[self.bc] = self.bcdiagval
                mat += sps.diags(diag, shape=mat.shape)

        if self.add_constant is not None:
            mat += self.add_constant
        return mat

    def _sensitivity(self, dgdmat: Union[DyadCarrier, np.ndarray]):
        if dgdmat.size <= 0:
            return [None]
        if self.bc is not None:
            dgdmat[self.bc, :] = 0.0
            dgdmat[:, self.bc] = 0.0
        dx = [np.zeros_like(s.state) for s in self.sig_in]
        if isinstance(dgdmat, np.ndarray):
            for i in range(self.nel):
                indu, indv = np.meshgrid(self.dofconn_row[i], self.dofconn_col[i], indexing="ij")
                for j in range(self.nmat):
                    dxi = einsum("ij,ij->", self.elmat[j], dgdmat[indu, indv])
                    dx[j][i] = np.real(dxi) if np.isrealobj(dx[j]) else dxi
        elif isinstance(dgdmat, DyadCarrier):
            for j in range(self.nmat):
                dxi = dgdmat.contract(self.elmat[j], self.dofconn_row, self.dofconn_col)
                dx[j][:] = np.real(dxi) if np.isrealobj(dx[j]) else dxi
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

        nnode = 2**domain.dim  # Number of nodes per element
        ndof = nnode * domain.dim

        # Element stiffness matrix
        dtype = np.result_type(D, domain.element_size.dtype)
        self.stiffness_element = np.zeros((ndof, ndof), dtype=dtype)

        # Numerical integration
        siz = domain.element_size
        w = np.prod(siz[: domain.dim] / 2)
        # Convert from area integral to volume in 2D
        if domain.dim == 2:
            w *= domain.element_size[2]

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
