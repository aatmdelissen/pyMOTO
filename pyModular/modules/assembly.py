""" Assembly modules for finite element analysis """
from pyModular import Module, DyadCarrier
import numpy as np
from scipy.sparse import csc_matrix


class DomainDefinition:
    """ Generic definitions for structured 2D or 3D domain """
    def __init__(self, nx: int, ny: int, nz: int = 0):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.dim = 2 if self.nz == 0 or self.nz is None else 3

        self.nel = self.nx * self.ny * max(self.nz, 1)  # Number of elements
        self.nnodes = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)  # Number of nodes

        self.elemnodes = 2 ** self.dim  # Number of dofs in each element

        elx = np.repeat(np.arange(self.nx), self.ny * max(self.nz, 1))
        ely = np.tile(np.repeat(np.arange(self.ny), max(self.nz, 1)), self.nx)
        elz = np.tile(np.arange(max(self.nz, 1)), self.nx * self.ny)
        el = self.get_elemnumber(elx, ely, elz)

        self.conn = np.zeros((self.nel, self.elemnodes), dtype=int)
        self.conn[el, :] = self.get_elemconnectivity(elx, ely, elz)

    def get_elemnumber(self, eli, elj, elk=0):
        return (elk * self.ny + elj) * self.nx + eli

    def get_nodenumber(self, nodi, nodj, nodk=0):
        return (nodk * (self.ny + 1) + nodj) * (self.nx + 1) + nodi

    def get_elemconnectivity(self, i, j, k=0):
        n1 = self.get_nodenumber(i,     j,     k)  # Node 1 number
        n2 = self.get_nodenumber(i + 1, j,     k)  # Node 2 number
        n3 = self.get_nodenumber(i,     j + 1, k)  # ...
        n4 = self.get_nodenumber(i + 1, j + 1, k)
        if self.dim == 2:
            return np.stack([n1, n2, n3, n4], axis=-1)
        elif self.dim == 3:
            n5 = self.get_nodenumber(i,     j,     k + 1)
            n6 = self.get_nodenumber(i + 1, j,     k + 1)
            n7 = self.get_nodenumber(i,     j + 1, k + 1)
            n8 = self.get_nodenumber(i + 1, j + 1, k + 1)
            return np.stack([n1, n2, n3, n4, n5, n6, n7, n8], axis=-1)
        else:
            raise ValueError("Unsupported number of dimensions {}".format(self.dim))

    def get_dofconnectivity(self, ndof: int):
        return np.repeat(self.conn*ndof, ndof, axis=-1) + np.tile(np.arange(ndof), self.elemnodes)


class AssembleGeneral(Module):
    def _prepare(self, domain: DomainDefinition, element_matrix: np.ndarray, bc=None, bcdiagval=1.0, *args, **kwargs):
        self.elmat = element_matrix
        self.ndof = self.elmat.shape[-1] // domain.elemnodes  # Number of dofs per node
        self.n = self.ndof * domain.nnodes  # Matrix size
        self.nnode = domain.nnodes

        self.dofconn = domain.get_dofconnectivity(self.ndof)

        # Row and column indices for the matrix
        self.rows = np.kron(self.dofconn, np.ones((domain.elemnodes*self.ndof, 1), dtype=int)).flatten()
        self.cols = np.kron(self.dofconn, np.ones((1, domain.elemnodes*self.ndof), dtype=int)).flatten()

        # Boundary conditions
        self.bc = bc
        self.bcdiagval = bcdiagval
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
            mat_values = np.concatenate((scaled_el[self.bcselect], self.bcdiagval*np.ones(len(self.bc))))
        else:
            mat_values = scaled_el

        mat = csc_matrix((mat_values, (self.rows, self.cols)), shape=(self.n, self.n))

        return mat

    def _sensitivity(self, dgdmat: DyadCarrier):
        return dgdmat.contract(self.elmat, self.dofconn, self.dofconn)


class AssembleStiffness(AssembleGeneral):
    """ Stiffness matrix assembly by scaling elements
    K = sum xScale_e K_e
    """
    def _prepare(self, domain: DomainDefinition, e_modulus: float = 1.0, poisson_ratio: float = 0.3, *args, **kwargs):
        """
        :param e_modulus: Base Young's modulus
        :param poisson_ratio: Base Poisson's ratio
        """
        self.E = e_modulus
        self.nu = poisson_ratio

        # Element stiffness matrix
        ly = 1.0
        lx = 1.0
        lz = 1.0

        c = ly / lx
        ka = (4 * c) * (1 - self.nu)
        kc = (4 / c) * (1 - self.nu)
        kd = (2 * c) * (1 - 2 * self.nu)
        kb = (2 / c) * (1 - 2 * self.nu)
        ke = self.E * lz / (12 * (1 + self.nu) * (1 - 2 * self.nu))
        kf = 6 * self.nu - 3 / 2

        self.KE = ke * np.array([
            [ka + kb, -3 / 2, ka / 2 - kb, kf, -ka + kb / 2, -kf, -ka / 2 - kb / 2, 3 / 2],
            [-3 / 2, kc + kd, -kf, -kc + kd / 2, kf, kc / 2 - kd, 3 / 2, -kc / 2 - kd / 2],
            [ka / 2 - kb, -kf, ka + kb, 3 / 2, -ka / 2 - kb / 2, -3 / 2, -ka + kb / 2, kf],
            [kf, -kc + kd / 2, 3 / 2, kc + kd, -3 / 2, -kc / 2 - kd / 2, -kf, kc / 2 - kd],
            [-ka + kb / 2, kf, -ka / 2 - kb / 2, -3 / 2, ka + kb, 3 / 2, ka / 2 - kb, -kf],
            [-kf, kc / 2 - kd, -3 / 2, -kc / 2 - kd / 2, 3 / 2, kc + kd, kf, -kc + kd / 2],
            [-ka / 2 - kb / 2, 3 / 2, -ka + kb / 2, -kf, ka / 2 - kb, kf, ka + kb, -3 / 2],
            [3 / 2, -kc / 2 - kd / 2, kf, kc / 2 - kd, -kf, -kc + kd / 2, -3 / 2, kc + kd],
        ])
        super()._prepare(domain, self.KE, *args, **kwargs)
