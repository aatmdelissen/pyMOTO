""" Assembly modules for finite element analysis
Assuming node numbering:

 3 -------- 4
 |          |
 |          |
 1 -------- 2

"""
import sys
from pymodular import Module, DyadCarrier
import numpy as np
from scipy.sparse import csc_matrix
from typing import Union

class DomainDefinition:
    """ Generic definitions for structured 2D or 3D domain
    Nodal numbering:
    Quadrangle in 2D
           ^
           | y
     3 -------- 4
     |     |    |   x
     |      --- | ---->
     1 -------- 2

    and in 3D Hexahedron:
           y
    2----------3
    |\     ^   |\
    | \    |   | \
    |  \   |   |  \
    |   6------+---7
    |   |  +-- |-- | -> x
    0---+---\--1   |
     \  |    \  \  |
      \ |     \  \ |
       \|      z  \|
        4----------5

    """

    def __init__(self, nx: int, ny: int, nz: int = 0, unitx: float = 1.0, unity: float = 1.0, unitz: float = 1.0):
        """ Creates a domain definition object of a structured mesh

        :param nx: Number of elements in x-direction
        :param ny: Number of elements in y-direction
        :param nz: (Optional) Number of elements in z-direction; if zero it is a 2D model
        :param unitx: Element size in x-direction
        :param unity: Element size in y-direction
        :param unitz: Element size in z-direction
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.dim = 2 if self.nz == 0 or self.nz is None else 3

        self.element_size = np.array([unitx, unity, unitz])
        assert np.prod(self.element_size[:self.dim]) > 0.0, 'Element volume needs to be positive'

        self.nel = self.nx * self.ny * max(self.nz, 1)  # Total number of elements
        self.nnodes = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)  # Total number of nodes

        self.elemnodes = 2 ** self.dim  # Number of nodes in each element

        # This is where the node numbering is defined, users may override this in their program to use custom numbering
        self.node_numbering = [[0, 0, 0] for _ in range(self.elemnodes)]
        self.node_numbering[0] = [-1, -1, -1]
        self.node_numbering[1] = [+1, -1, -1]
        if self.dim >= 2:
            self.node_numbering[2] = [-1, +1, -1]
            self.node_numbering[3] = [+1, +1, -1]
        if self.dim >= 3:
            self.node_numbering[4] = [-1, -1, +1]
            self.node_numbering[5] = [+1, -1, +1]
            self.node_numbering[6] = [-1, +1, +1]
            self.node_numbering[7] = [+1, +1, +1]

        # Get the element numbers
        elx = np.repeat(np.arange(self.nx), self.ny * max(self.nz, 1))
        ely = np.tile(np.repeat(np.arange(self.ny), max(self.nz, 1)), self.nx)
        elz = np.tile(np.arange(max(self.nz, 1)), self.nx * self.ny)
        el = self.get_elemnumber(elx, ely, elz)

        # Setup node-element connectivity
        self.conn = np.zeros((self.nel, self.elemnodes), dtype=int)
        self.conn[el, :] = self.get_elemconnectivity(elx, ely, elz)

    def get_elemnumber(self, eli: Union[int, np.ndarray], elj: Union[int, np.ndarray], elk: Union[int, np.ndarray] = 0):
        return (elk * self.ny + elj) * self.nx + eli

    def get_nodenumber(self, nodi: Union[int, np.ndarray], nodj: Union[int, np.ndarray], nodk: Union[int, np.ndarray] = 0):
        return (nodk * (self.ny + 1) + nodj) * (self.nx + 1) + nodi

    def get_elemconnectivity(self, i: Union[int, np.ndarray], j: Union[int, np.ndarray], k: Union[int, np.ndarray] = 0):
        """ Get the connectivity for element identified with cartesian indices (i, j, k)
        This is where the nodal numbers are defined
        :param i: Ith element in the x-direction; can be integer or array
        :param j: Jth element in the y-direction; can be integer or array
        :param k: Kth element in the z-direction; can be integer or array
        :return: The node numbers corresponding to selected elements
        """
        nods = [self.get_nodenumber(i+max(n[0], 0), j+max(n[1], 0), k+max(n[2], 0)) for n in self.node_numbering]
        return np.stack(nods, axis=-1)

    def get_dofconnectivity(self, ndof: int):
        return np.repeat(self.conn*ndof, ndof, axis=-1) + np.tile(np.arange(ndof), self.elemnodes)

    def eval_shape_fun(self, pos: np.ndarray):
        """
        In 1D (bar):
           N1 = 1/w (w/2 - x)
           N2 = 1/w (w/2 + x)

        Shape functions: Cook eq. (6.2-3)
           N1 = 1/(wh) (w/2 - x) (h/2 - y)
           N2 = 1/(wh) (w/2 + x) (h/2 - y)
           N3 = 1/(wh) (w/2 - x) (h/2 + y)
           N4 = 1/(wh) (w/2 + x) (h/2 + y)

        In 3D:
           N1 = 1/(whd) (w/2 - x) (h/2 - y) (d/2 - z)
           ...

        :param pos: Evaluation point, [x, y, z (optional)] - coordinates
        :param element_size: Element dimensions in x, y, and z directions [w, h, d (optional)]
        :return: Array of evaluated shape functions [N1(x,y,z), N2(x,y,z), ...]
        """
        v = np.prod(self.element_size[:self.dim])
        assert v > 0.0, 'Element volume needs to be positive'
        ret = np.ones(self.nnodes)/v
        for i in range(self.dim):
            ret *= np.array([self.element_size[i] + n[i]*pos[i] for n in self.node_numbering])
        return ret

    def eval_shape_fun_der(self, pos: np.ndarray):
        """ Evaluates the shape function derivatives in x, y, and optionally z-direction.
        For 1D domains, the y and z directions are optional.
        For 2D domains, the z direction is optional.
        :param pos: Evaluation point, [x, y, z] - element coordinates in intervals [-w/2, w/2], [-h/2, h/2], [-d/2, d/2]
        :param element_size: Element dimensions in x, y, and z directions [w, h, d]
        :return: Shape function derivatives of size (#dimensions, #shape functions)
        """
        v = np.prod(self.element_size[:self.dim])
        assert v > 0.0, 'Element volume needs to be positive'
        dN_dx = np.ones((self.dim, self.elemnodes))/v  # dN/dx = 1/V
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j: # dN/dx_i *= (w[j]/2 ± x[j])
                    dN_dx[i, :] *= np.array([self.element_size[j]/2 + n[j]*pos[j] for n in self.node_numbering])
            dN_dx[i, :] *= np.array([n[i] for n in self.node_numbering])  # Flip +/- signs according to node position
        return dN_dx


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
            raise type(e)(str(e) + "\n\tInvalid matrix_type={}. Use either scipy.sparse.cscmatrix or scipy.sparse.csrmatrix".format(self.matrix_type)).with_traceback(sys.exc_info()[2]) from None

        return mat

    def _sensitivity(self, dgdmat: Union[DyadCarrier, np.ndarray]):
        if self.bc is not None:
            dgdmat[self.bc, :] = 0.0
            dgdmat[:, self.bc] = 0.0
        if isinstance(dgdmat, np.ndarray):
            dx = np.zeros_like(self.sig_in[0].state)
            for i in range(len(dx)):
                indu, indv = np.meshgrid(self.dofconn[i], self.dofconn[i], indexing='ij')
                dx[i] = np.einsum("ij,ij->", self.elmat, dgdmat[indu, indv])
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
    n_strains = int((n_dim * (n_dim+1))/2) # Triangular number: ndim=3 -> nstrains = 3+2+1
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
    """ Stiffness matrix assembly by scaling elements ( Plane stress )
    K = sum xScale_e K_e
    """
    def _prepare(self, domain: DomainDefinition, *args, e_modulus: float = 1.0, poisson_ratio: float = 0.3, plane='strain', **kwargs):
        """
        :param e_modulus: Base Young's modulus
        :param poisson_ratio: Base Poisson's ratio
        :param lx: Element size in x-direction
        :param ly: Element size in y-direction
        :param lz: Element size in z-direction (thickness)
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
    def _prepare(self, domain: DomainDefinition, *args, rho: float = 1.0, **kwargs):
        """
        :param rho: Base density
        :param lx: Element size in x-direction
        :param ly: Element size in y-direction
        :param lz: Element size in z-direction (thickness)
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
        super()._prepare(domain, ME, *args, **kwargs)