import os
import sys
import base64
import struct
import warnings
from typing import Union
import numpy as np


class DomainDefinition:
    r""" Definition for a structured domain
    Nodal numbering used in the domain is given below.

    Quadrangle in 2D

    ::

              ^
              | y
        3 -------- 4
        |     |    |   x
        |      --- | ---->
        |          |
        1 -------- 2


    Hexahedron in 3D

    ::

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


    Args:
        nelx : Number of elements in x-direction
        nely : Number of elements in y-direction
        nelz : Number of elements in z-direction; if zero it is a 2D domain
        unitx : Element size in x-direction
        unity : Element size in y-direction
        unitz : Element size in z-direction

    Attributes:
        dim : Dimensionality of the object
        nel : Total number of elements
        nnodes : Total number of nodes
        elemnodes : Number of nodes per element
        node_numbering : The numbering scheme used to number the nodes in each element
        conn : Connectivity matrix of size (# elements, # nodes per element)
    """

    def __init__(self, nelx: int, nely: int, nelz: int = 0, unitx: float = 1.0, unity: float = 1.0, unitz: float = 1.0):
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        if self.nely is None:
            self.nely = 0
        if self.nelz is None:
            self.nelz = 0
        self.unitx, self.unity, self.unitz = unitx, unity, unitz

        self.dim = 1 if (self.nelz == 0 and self.nely == 0) else (2 if self.nelz == 0 else 3)

        self.element_size = np.array([unitx, unity, unitz])
        assert np.prod(self.element_size[:self.dim]) > 0.0, 'Element volume needs to be positive'

        self.nel = self.nelx * self.nely * max(self.nelz, 1)  # Total number of elements
        self.nnodes = (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)  # Total number of nodes

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
        elx = np.repeat(np.arange(self.nelx), self.nely * max(self.nelz, 1))
        ely = np.tile(np.repeat(np.arange(self.nely), max(self.nelz, 1)), self.nelx)
        elz = np.tile(np.arange(max(self.nelz, 1)), self.nelx * self.nely)
        el = self.get_elemnumber(elx, ely, elz)

        # Setup node-element connectivity
        self.conn = np.zeros((self.nel, self.elemnodes), dtype=int)
        self.conn[el, :] = self.get_elemconnectivity(elx, ely, elz)

    def get_elemnumber(self, eli: Union[int, np.ndarray], elj: Union[int, np.ndarray], elk: Union[int, np.ndarray] = 0):
        """ Gets the element number(s) for element(s) with given Cartesian indices (i, j, k)

        Args:
            eli : Ith element in the x-direction; can be integer or array
            elj : Jth element in the y-direction; can be integer or array
            elk : Kth element in the z-direction; can be integer or array

        Returns:
            The element number(s) corresponding to selected indices
        """
        return (elk * self.nely + elj) * self.nelx + eli

    def get_nodenumber(self, nodi: Union[int, np.ndarray], nodj: Union[int, np.ndarray], nodk: Union[int, np.ndarray] = 0):
        """ Gets the node number(s) for nodes with given Cartesian indices (i, j, k)

        Args:
            nodi : Ith node in the x-direction; can be integer or array
            nodj : Jth node in the y-direction; can be integer or array
            nodk : Kth node in the z-direction; can be integer or array

        Returns:
            The node number(s) corresponding to selected indices
        """
        return (nodk * (self.nely + 1) + nodj) * (self.nelx + 1) + nodi

    def get_node_indices(self, nod_idx: Union[int, np.ndarray]):
        """ Gets the Cartesian index (i, j, k) for given node number(s)

        Args:
            nod_idx: Node index; can be integer or array

        Returns:
            i, j, k for requested node(s); k is only returned in 3D
        """
        nodi = nod_idx % (self.nelx + 1)
        nodj = (nod_idx // (self.nelx + 1)) % (self.nely + 1)
        if self.dim == 2:
            return nodi, nodj
        nodk = nod_idx // ((self.nelx + 1)*(self.nely + 1))
        return nodi, nodj, nodk

    def get_node_position(self, nod_idx: Union[int, np.ndarray]):
        ijk = self.get_node_indices(nod_idx)
        return [idx * self.element_size[ii] for ii, idx in enumerate(ijk)]

    def get_elemconnectivity(self, i: Union[int, np.ndarray], j: Union[int, np.ndarray], k: Union[int, np.ndarray] = 0):
        """ Get the connectivity for element identified with Cartesian indices (i, j, k)
        This is where the nodal numbers are defined

        Args:
            i: Ith element in the x-direction; can be integer or array
            j: Jth element in the y-direction; can be integer or array
            k: Kth element in the z-direction; can be integer or array

        Returns:
            The node numbers corresponding to selected elements of size (# selected elements, # nodes per element)
        """
        nods = [self.get_nodenumber(i+max(n[0], 0), j+max(n[1], 0), k+max(n[2], 0)) for n in self.node_numbering]
        return np.stack(nods, axis=-1)

    def get_dofconnectivity(self, ndof: int):
        """ Get the connectivity in terms of degrees of freedom

        Args:
            ndof: The number of degrees of freedom per node

        Returns:
            The dof numbers corresponding to each element of size (# total elements, # dofs per element)
        """
        return np.repeat(self.conn*ndof, ndof, axis=-1) + np.tile(np.arange(ndof), self.elemnodes)

    def eval_shape_fun(self, pos: np.ndarray):
        r""" Evaluate the linear shape functions of the finite element

        In 1D
          * :math:`N_1(x) = \frac{1}{w} \left(\frac{w}{2} - x\right)`
          * :math:`N_2(x) = \frac{1}{w} \left(\frac{w}{2} + x\right)`

        In 2D [1]
          * :math:`N_1(x,y) = \frac{1}{A} \left(\frac{w}{2} - x\right) \left(\frac{h}{2} - y\right)`
          * :math:`N_2(x,y) = \frac{1}{A} \left(\frac{w}{2} + x\right) \left(\frac{h}{2} - y\right)`
          * :math:`N_3(x,y) = \frac{1}{A} \left(\frac{w}{2} - x\right) \left(\frac{h}{2} + y\right)`
          * :math:`N_4(x,y) = \frac{1}{A} \left(\frac{w}{2} + x\right) \left(\frac{h}{2} + y\right)`

        with :math:`A = wh`

        In 3D
          * :math:`N_1(x,y,z) = \frac{1}{V} \left(\frac{w}{2} - x\right) \left(\frac{h}{2} - y\right) \left(\frac{d}{2} - z\right)`
          * etc.

        with :math:`V = whd`

        Args:
            pos : Evaluation coordinates [x, y, z (optional)] within bounds of [-element_size/2, element_size/2]

        Returns:
            Array of evaluated shape functions [N1(x), N2(x), ...]

        References:
            [1] Cook, Malkus, Plesha, Witt (2002). Concepts and applications of finite element analysis (4th ed.), eq. (6.2-3)
        """
        v = np.prod(self.element_size[:self.dim])
        assert v > 0.0, 'Element volume needs to be positive'
        shapefn = np.ones(self.elemnodes)/v
        for i in range(self.dim):
            shapefn *= np.array([self.element_size[i]/2 + n[i]*pos[i] for n in self.node_numbering])
        return shapefn

    def eval_shape_fun_der(self, pos: np.ndarray):
        """ Evaluates the shape function derivatives in x, y, and optionally z-direction.
        For 1D domains, the y and z directions are optional.
        For 2D domains, the z direction is optional.

        Args:
            pos : Evaluation coordinates [x, y, z(optional)] within bounds of [-element_size/2, element_size/2]

        Returns:
            Shape function derivatives of size (#dimensions, #shape functions)
        """
        v = np.prod(self.element_size[:self.dim])
        assert v > 0.0, 'Element volume needs to be positive'
        dN_dx = np.ones((self.dim, self.elemnodes))/v  # dN/dx = 1/V
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:  # dN/dx_i *= (w[j]/2 ± x[j])
                    dN_dx[i, :] *= np.array([self.element_size[j]/2 + n[j]*pos[j] for n in self.node_numbering])
            dN_dx[i, :] *= np.array([n[i] for n in self.node_numbering])  # Flip +/- signs according to node position
        return dN_dx

    # flake8: noqa: C901
    def write_to_vti(self, vectors: dict, filename="out.vti", scale=1.0, origin=(0.0, 0.0, 0.0)):
        """ Write all given vectors to a Paraview (VTI) file

        The size of the vectors should be a multiple of ``nel`` or ``nnodes``. Based on their size they are marked as
        cell-data or point-data in the VTI file. For 2D data (size is equal to ``2*nnodes``), the z-dimension is padded
        with zeros to have 3-dimensional data. Also block-vectors of multiple dimensions (*e.g.* ``(2, 3*nnodes)``) are
        accepted, which get the suffixed as ``_00``.

        Args:
            vectors: A dictionary of vectors to write. Keys are used as vector names.
            filename (str): The file loction
            scale: Uniform scaling of the gridpoints
            origin: Origin of the domain
        """
        ext = '.vti'
        if ext not in os.path.splitext(filename)[-1].lower():
            filename += ext

        # Sort into point-data and cell-data
        point_dat = {}
        cell_dat = {}
        for key, vec in vectors.items():
            if vec.size % self.nel == 0:
                cell_dat[key] = vec
            elif vec.size % self.nnodes == 0:
                point_dat[key] = vec
            else:
                warnings.warn(f"Vector {key} is neither cell- nor point-data. Skipping vector...")

        if len(point_dat) == 0 and len(cell_dat) == 0:
            warnings.warn(f"Nothing to write to {filename}. Skipping file...")
            return

        len_enc = ('<' if sys.byteorder == 'little' else '>') + 'Q'

        with open(filename, 'wb') as file:
            # XML header
            file.write(b'<?xml version=\"1.0\"?>\n')

            # Vtk header
            byte_order = "LittleEndian" if sys.byteorder == "little" else "BigEndian"
            file.write(f'<VTKFile type=\"ImageData\" version=\"0.1\" '
                       f'header_type=\"UInt64\" '
                       f'byte_order=\"{byte_order}\">\n'.encode())

            # Extend of coordinates
            file.write(f"<ImageData WholeExtent=\"0 {self.nelx} 0 {self.nely} 0 {self.nelz}\"".encode())

            # Origin of domain
            file.write(f" Origin=\"{origin[0]*scale} {origin[1]*scale} {origin[2]*scale}\"".encode())

            # Spacing of points (dx, dy, dz)
            dx, dy, dz = self.element_size[0:3]*scale
            file.write(f" Spacing=\"{dx} {dy} {dz}\">\n".encode())

            # Start new piece
            file.write(f"<Piece Extent=\"0 {self.nelx} 0 {self.nely} 0 {self.nelz}\">\n".encode())

            # Start writing pointdata
            if len(point_dat) > 0:
                file.write(b'<PointData>\n')
                for key, vec in point_dat.items():
                    vecax = next((i for i, s in enumerate(vec.shape) if s % self.nnodes == 0), None)
                    ncomponents = vec.shape[vecax]//self.nnodes
                    # Vectorize 2D vectors, by padding with 0's
                    pad_to_vector = ncomponents == 2 and self.dim == 2
                    assert vec.ndim <= 2, "Only for 1D and 2D numpy arrays"
                    nvectors = 1 if vec.ndim == 1 else vec.shape[(vecax+1) % 2]
                    nzeros = int(np.ceil(np.log10(nvectors)))
                    for i in range(nvectors):
                        vecname = key
                        if nvectors > 1:
                            intstr = i.__format__(f'0{nzeros}d')
                            vecname += f"({intstr})"
                            ind = [slice(None), slice(None)]
                            ind[(vecax+1) % 2] = i
                            vec_to_write = vec[tuple(ind)].astype(np.float32)
                        else:
                            vec_to_write = vec.astype(np.float32)

                        if pad_to_vector:
                            vec_pad = np.zeros(3*self.nnodes, dtype=np.float32)
                            vec_pad[0::3] = vec_to_write[0::2]
                            vec_pad[1::3] = vec_to_write[1::2]
                            vec_to_write = vec_pad

                        file.write(f'<DataArray type=\"Float32\" '
                                   f'Name=\"{vecname}\" '
                                   f'NumberOfComponents=\"{3 if pad_to_vector else ncomponents}\" '
                                   f'format=\"binary\">\n'.encode())
                        enc_data = base64.b64encode(vec_to_write)  # Encode the data
                        # Get the length of encoded data block
                        enc_len = base64.b64encode(struct.pack(len_enc, len(enc_data)))
                        file.write(enc_len)  # Write length
                        file.write(enc_data)  # Write data
                        file.write(b'\n</DataArray>\n')
                file.write(b'</PointData>\n')

            # Start writing celldata
            if len(cell_dat) > 0:
                file.write(b'<CellData>\n')
                for key, vec in cell_dat.items():
                    vecax = next((i for i, s in enumerate(vec.shape) if s % self.nel == 0), None)
                    ncomponents = vec.shape[vecax] // self.nel
                    assert vec.ndim <= 2, "Only for 1D and 2D numpy arrays"
                    nvectors = 1 if vec.ndim == 1 else vec.shape[(vecax+1) % 2]

                    for i in range(nvectors):
                        vecname = key
                        if nvectors > 1:
                            vecname += f"({i})"
                            ind = [slice(None), slice(None)]
                            ind[(vecax+1) % 2] = i
                            vec_to_write = vec[tuple(ind)].astype(np.float32)
                        else:
                            vec_to_write = vec.astype(np.float32)

                        file.write(f'<DataArray type=\"Float32\" '
                                   f'Name=\"{vecname}\" '
                                   f'NumberOfComponents=\"{ncomponents}\" '
                                   f'format=\"binary\">\n'.encode())
                        enc_data = base64.b64encode(vec_to_write)  # Encode the data
                        # Get the length of encoded data block
                        enc_len = base64.b64encode(struct.pack(len_enc, len(enc_data)))
                        file.write(enc_len)  # Write length
                        file.write(enc_data)  # Write data
                        file.write(b'\n</DataArray>\n')
                file.write(b'</CellData>\n')

            file.write(b'</Piece>\n')
            file.write(b'</ImageData>\n')
            file.write(b'</VTKFile>')
