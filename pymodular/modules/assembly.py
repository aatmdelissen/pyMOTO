""" Assembly modules for finite element analysis
Assuming node numbering:

 3 -------- 4
 |          |
 |          |
 1 -------- 2

"""
import sys
import base64  # For binary writing
import struct  # For binary writing
import os.path
from pymodular import Module, DyadCarrier, DomainDefinition
import numpy as np
from scipy.sparse import csc_matrix
from typing import Union
import warnings
try:
    from opt_einsum import contract as einsum
except ModuleNotFoundError:
    from numpy import einsum


class AssembleGeneral(Module):
    """ Assembles a sparse matrix according to element scaling
    """
    def _prepare(self, domain: DomainDefinition, element_matrix: np.ndarray, bc=None, bcdiagval=None, matrix_type=csc_matrix):
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
            raise type(e)(str(e) + "\n\tInvalid matrix_type={}. Either scipy.sparse.cscmatrix or scipy.sparse.csrmatrix are supported".format(self.matrix_type)).with_traceback(sys.exc_info()[2]) from None

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
    1D : [ε_x]_i = B [u]_i
    2D : [ε_x; ε_y; γ_xy]_i = B [u, v]_i
    3D : [ε_x; ε_y; ε_z; γ_xy; γ_yz; γ_zx]_i = B [u, v, w]_i
    :param dN_dx: Shape function derivatives [dNi_dxj] of size (#shapefn. x #dimensions)
    :return: B strain-displacement relation of size (#strains x #shapefn.*#dimensions)
    """
    n_dim, n_shapefn = dN_dx.shape
    n_strains = int((n_dim * (n_dim+1))/2)  # Triangular number: ndim=3 -> nstrains = 3+2+1
    B = np.zeros((n_strains, n_shapefn*n_dim))
    if n_dim == 1:
        for i in range(n_shapefn):
            B[i, 0] = dN_dx[i, 0]
    elif n_dim == 2:
        for i in range(n_shapefn):
            B[:, i*n_dim:(i+1)*n_dim] = np.array([[dN_dx[0, i], 0          ],
                                                  [0,           dN_dx[1, i]],
                                                  [dN_dx[1, i], dN_dx[0, i]]])
    elif n_dim == 3:
        for i in range(n_shapefn):
            B[:, i*n_dim:(i+1)*n_dim] = np.array([[dN_dx[0, i], 0,           0          ],
                                                  [0,           dN_dx[1, i], 0          ],
                                                  [0,           0,           dN_dx[2, i]],
                                                  [dN_dx[1, i], dN_dx[0, i], 0          ],
                                                  [0,           dN_dx[2, i], dN_dx[1, i]],
                                                  [dN_dx[2, i], 0,           dN_dx[0, i]]])
    else:
        raise ValueError(f"Number of dimensions ({n_dim}) cannot be greater than 3")
    return B


def get_D(E, nu, mode='strain'):
    """ Get material constitutive relation for linear elasticity

    :param E: Young's modulus
    :param nu: Poisson's ratio
    :param mode: Plane-"strain", plane-"stress", or "3D"
    :return: Material matrix
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
    """ Stiffness matrix assembly by scaling elements in 2D or 3D
    K = sum xScale_e K_e
    """
    def _prepare(self, domain: DomainDefinition, *args, e_modulus: float = 1.0, poisson_ratio: float = 0.3, plane='strain', **kwargs):
        """
        :param domain: The domain to assemble for -- this determines the element size and dimensionality
        :param args: Other arguments of AssembleGeneral
        :param e_modulus: Young's modulus
        :param poisson_ratio: Poisson's ratio
        :param plane: The 2D in-plane condition: "strain" or "stress" for rsp. plane-strain or plane-stress
        :param kwargs: Other keyword-arguments of AssembleGeneral
        """
        self.E, self.nu = e_modulus, poisson_ratio

        # Get material relation
        D = get_D(self.E, self.nu, '3d' if domain.dim == 3 else plane.lower())
        if domain.dim == 2:
            D *= domain.element_size[2]

        nnode = 2**domain.dim  # Number of nodes per element
        ndof = nnode*domain.dim

        # Element stiffness matrix
        self.KE = np.zeros((ndof, ndof))

        # Numerical integration
        siz = domain.element_size
        w = np.prod(siz[:domain.dim]/2)
        for n in domain.node_numbering:
            pos = n*(siz/2)/np.sqrt(3)  # Sampling point
            dN_dx = domain.eval_shape_fun_der(pos)
            B = get_B(dN_dx)
            self.KE += w * B.T @ D @ B  # Add contribution

        super()._prepare(domain, self.KE, *args, **kwargs)


class AssembleMass(AssembleGeneral):
    """ Consistent mass matrix assembly by scaling elements
    M = sum xScale_e M_e
    """
    def _prepare(self, domain: DomainDefinition, *args, rho: float = 1.0, bcdiagval=0.0, **kwargs):
        """
        :param domain: The domain to assemble for -- this determines the element size and dimensionality
        :param args: Other arguments of AssembleGeneral
        :param rho: Base density
        :param bcdiagval: The value to put on the diagonal in case of boundary conditions (bc)
        :param kwargs: Other keyword-arguments of AssembleGeneral
        """

        # Element mass matrix
        # 1/36 Mass of one element
        mel = rho * np.prod(domain.element_size)

        # Consistent mass matrix
        ME = mel / 36 * np.array([[4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0],
                                  [0.0, 4.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0],
                                  [2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                                  [0.0, 2.0, 0.0, 4.0, 0.0, 1.0, 0.0, 2.0],
                                  [2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 2.0, 0.0],
                                  [0.0, 2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 2.0],
                                  [1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0, 0.0],
                                  [0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0]])

        super()._prepare(domain, ME, *args, bcdiagval=bcdiagval, **kwargs)
