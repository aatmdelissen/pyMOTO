from typing import Union
import numpy as np


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

    def __init__(self, nelx: int, nely: int, nelz: int = 0, unitx: float = 1.0, unity: float = 1.0, unitz: float = 1.0):
        """ Creates a domain definition object of a structured mesh

        :param nelx: Number of elements in x-direction
        :param nely: Number of elements in y-direction
        :param nelz: (Optional) Number of elements in z-direction; if zero it is a 2D model
        :param unitx: Element size in x-direction
        :param unity: Element size in y-direction
        :param unitz: Element size in z-direction
        """
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        self.unitx, self.unity, self.unitz = unitx, unity, unitz

        self.dim = 2 if self.nelz == 0 or self.nelz is None else 3

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
        return (elk * self.nely + elj) * self.nelx + eli

    def get_nodenumber(self, nodi: Union[int, np.ndarray], nodj: Union[int, np.ndarray], nodk: Union[int, np.ndarray] = 0):
        return (nodk * (self.nely + 1) + nodj) * (self.nelx + 1) + nodi

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
                if i != j:  # dN/dx_i *= (w[j]/2 ± x[j])
                    dN_dx[i, :] *= np.array([self.element_size[j]/2 + n[j]*pos[j] for n in self.node_numbering])
            dN_dx[i, :] *= np.array([n[i] for n in self.node_numbering])  # Flip +/- signs according to node position
        return dN_dx

    def write_to_vti(self, vectors: dict, filename="out.vti", scale=1.0, origin=(0.0, 0.0, 0.0)):
        """ Write all given vectors to a Paraview (VTI) file
        :param vectors: A dictionary of vectors to write. Keys are used as vector names
        :param filename: The file loction
        :param scale: Uniform scaling of the gridpoints
        :param origin: Origin of the domain
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
            file.write(f'<VTKFile type=\"ImageData\" version=\"0.1\" header_type=\"UInt64\" byte_order=\"{"LittleEndian" if sys.byteorder=="little" else "BigEndian"}\">\n'.encode())

            # Extend of coordinates
            file.write(f"<ImageData WholeExtent=\"0 {self.nelx} 0 {self.nely} 0 {self.nelz}\"".encode())

            # Origin of domain
            file.write(f" Origin=\"{origin[0]*scale} {origin[1]*scale} {origin[2]*scale}\"".encode())

            # Spacing of points (dx, dy, dz)
            file.write(f" Spacing=\"{self.element_size[0]*scale} {self.element_size[1]*scale} {self.element_size[2]*scale}\">\n".encode())

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

                        file.write(f'<DataArray type=\"Float32\" Name=\"{vecname}\" NumberOfComponents=\"{3 if pad_to_vector else ncomponents}\" format=\"binary\">\n'.encode())
                        enc_data = base64.b64encode(vec_to_write)  # Encode the data
                        enc_len = base64.b64encode(struct.pack(len_enc, len(enc_data)))  # Get the length of encoded data block
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

                        file.write(f'<DataArray type=\"Float32\" Name=\"{vecname}\" NumberOfComponents=\"{ncomponents}\" format=\"binary\">\n'.encode())
                        enc_data = base64.b64encode(vec_to_write)  # Encode the data
                        enc_len = base64.b64encode(struct.pack(len_enc, len(enc_data)))  # Get the length of encoded data block
                        file.write(enc_len)  # Write length
                        file.write(enc_data)  # Write data
                        file.write(b'\n</DataArray>\n')
                file.write(b'</CellData>\n')

            file.write(b'</Piece>\n')
            file.write(b'</ImageData>\n')
            file.write(b'</VTKFile>')
