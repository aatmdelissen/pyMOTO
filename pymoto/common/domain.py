import os
import sys
import base64
import struct
import warnings
from typing import Union, Iterable, Protocol

from numpy.typing import NDArray
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from ..utils import _parse_to_list


def plot_deformed_element(ax, x, y, **kwargs):
    codes, verts = zip(
        *[
            (Path.MOVETO, [x[0], y[0]]),
            (Path.LINETO, [x[1], y[1]]),
            (Path.LINETO, [x[3], y[3]]),
            (Path.LINETO, [x[2], y[2]]),
            (Path.CLOSEPOLY, [x[0], y[0]]),
        ]
    )
    path = Path(verts, codes)
    patch = PathPatch(path, **kwargs)
    ax.add_artist(patch)
    return patch


def get_path(x, y):
    codes, verts = zip(
        *[
            (Path.MOVETO, [x[0], y[0]]),
            (Path.LINETO, [x[1], y[1]]),
            (Path.LINETO, [x[3], y[3]]),
            (Path.LINETO, [x[2], y[2]]),
            (Path.CLOSEPOLY, [x[0], y[0]]),
        ]
    )
    return Path(verts, codes)


IndexType = Union[int, Iterable[int], NDArray[np.integer]]

class MeshT(Protocol):
    """General mesh type, modeled after https://github.com/wolph/numpy-stl """
    min_: np.ndarray
    max_: np.ndarray
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    name: str
    

class VoxelDomain:
    r""" Definition for a structured voxel domain
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

    Attributes:
        dim : Dimensionality of the object
        nel : Total number of elements
        nnodes : Total number of nodes
        elemnodes : Number of nodes per element
        node_numbering : The numbering scheme used to number the nodes in each element
        conn : Connectivity matrix of size (# elements, # nodes per element)
        elements : Helper array for element slicing of size (nelx, nely, nelz)
        nodes : Helper array for node slicing of size (nelx+1, nely+1, nelz+1)
    """

    def __init__(self, nelx: int, nely: int, nelz: int = 0, unitx: float = 1.0, unity: float = 1.0, unitz: float = 1.0):
        """Create a 2D or 3D voxel domain

        Args:
            nelx (int): Number of elements in x-direction
            nely (int): Number of elements in y-direction
            nelz (int, optional): Number of elements in z-direction; if zero it is a 2D domain. Defaults to 0.
            unitx (float, optional): Element size in x-direction. Defaults to 1.0.
            unity (float, optional): Element size in y-direction. Defaults to 1.0.
            unitz (float, optional): Element size in z-direction. Defaults to 1.0.
        """
        self.nelx, self.nely, self.nelz = nelx, nely, nelz
        if self.nely is None:
            self.nely = 0
        if self.nelz is None:
            self.nelz = 0

        self.dim = 1 if (self.nelz == 0 and self.nely == 0) else (2 if self.nelz == 0 else 3)

        self.origin = np.array([0.0, 0.0, 0.0])
        self.unitx, self.unity, self.unitz = unitx, unity, unitz

        assert np.prod(self.element_size[: self.dim]) > 0.0, "Element volume needs to be positive"

        self.nel = self.nelx * self.nely * max(self.nelz, 1)  # Total number of elements
        self.nnodes = (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)  # Total number of nodes

        self.elemnodes = 2**self.dim  # Number of nodes in each element

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

        # Helper for element slicing
        eli, elj, elk = np.meshgrid(
            np.arange(self.nelx), np.arange(self.nely), np.arange(max(self.nelz, 1)), indexing="ij"
        )
        self.elements = self.get_elemnumber(eli, elj, elk)

        # Helper for node slicing
        ndi, ndj, ndk = np.meshgrid(
            np.arange(self.nelx + 1), np.arange(self.nely + 1), np.arange(self.nelz + 1), indexing="ij"
        )
        self.nodes = self.get_nodenumber(ndi, ndj, ndk)

    @property
    def element_size(self):
        """Element size in each direction"""
        return np.array([self.unitx, self.unity, self.unitz])

    @property
    def domain_size(self):
        """Domain size in each direction"""
        return np.array([self.nelx * self.unitx, self.nely * self.unity, self.nelz * self.unitz])[: self.dim]

    @property
    def size(self):
        """Number of elements in each direction"""
        return np.array([self.nelx, self.nely, self.nelz])[: self.dim]

    def get_elemnumber(self, eli: IndexType, elj: IndexType, elk: IndexType = 0):
        """Gets the element number(s) for element(s) with given Cartesian indices (i, j, k)

        Args:
            eli : Ith element in the x-direction; can be integer or array
            elj : Jth element in the y-direction; can be integer or array
            elk : Kth element in the z-direction; can be integer or array

        Returns:
            The element number(s) corresponding to selected indices
        """
        return (elk * self.nely + elj) * self.nelx + eli

    def get_nodenumber(self, nodi: IndexType, nodj: IndexType, nodk: IndexType = 0):
        """Gets the node number(s) for nodes with given Cartesian indices (i, j, k)

        Args:
            nodi : Ith node in the x-direction; can be integer or array
            nodj : Jth node in the y-direction; can be integer or array
            nodk : Kth node in the z-direction; can be integer or array

        Returns:
            The node number(s) corresponding to selected indices
        """
        return (nodk * (self.nely + 1) + nodj) * (self.nelx + 1) + nodi

    def get_dofnumber(self, nod_idx: IndexType, dof_idx: IndexType = None, ndof: int = None):
        """Gets the degree of freedom number(s) for node(s) with given node number(s)

        Args:
            nod_idx : Node number; can be integer or array
            dof_idx (optional) : Dof index to request (e.g. `0` for x, `[0, 1]` for x and y) (default is all dofs)
            ndof (optional) : Number of degrees of freedom per node (default is `domain.dim`)

        Returns:
            The dof number(s) corresponding to selected node index(es)
        """
        if not isinstance(nod_idx, int):
            nod_idx = np.asarray(nod_idx)
        if ndof is None:
            ndof = self.dim
        if dof_idx is None:
            dof_idx = np.arange(ndof)
        if not isinstance(dof_idx, int):
            dof_idx = np.asarray(dof_idx)

        if np.ndim(dof_idx) == 0 or np.ndim(nod_idx) == 0:
            return nod_idx * ndof + dof_idx
        else:
            nod_idx1 = np.expand_dims(nod_idx, axis=tuple(-(np.arange(np.ndim(dof_idx)) + 1)))
            dof_idx1 = np.expand_dims(dof_idx, axis=tuple(np.arange(np.ndim(nod_idx))))
            return nod_idx1 * ndof + dof_idx1

    def get_node_indices(self, nod_idx: IndexType = None):
        """Gets the Cartesian index (i, j, k) for given node number(s)

        Args:
            nod_idx: Node index; can be integer or array

        Returns:
            i, j, k for requested node(s); k is only returned in 3D
        """
        if nod_idx is None:
            nod_idx = np.arange(self.nnodes)
        nodi = nod_idx % (self.nelx + 1)
        nodj = (nod_idx // (self.nelx + 1)) % (self.nely + 1)
        if self.dim == 2:
            return np.stack([nodi, nodj], axis=0)
        nodk = nod_idx // ((self.nelx + 1) * (self.nely + 1))
        return np.stack([nodi, nodj, nodk], axis=0)

    def get_node_position(self, nod_idx: IndexType = None):
        ijk = self.get_node_indices(nod_idx)
        return (self.element_size[: self.dim] * ijk.T).T

    def get_elemconnectivity(self, i: IndexType, j: IndexType, k: IndexType = 0):
        """Get the connectivity for element identified with Cartesian indices (i, j, k)
        This is where the nodal numbers are defined

        Args:
            i: Ith element in the x-direction; can be integer or array
            j: Jth element in the y-direction; can be integer or array
            k: Kth element in the z-direction; can be integer or array

        Returns:
            The node numbers corresponding to selected elements of size (# selected elements, # nodes per element)
        """
        nods = [self.get_nodenumber(i + max(n[0], 0), j + max(n[1], 0), k + max(n[2], 0)) for n in self.node_numbering]
        return np.stack(nods, axis=-1)

    def get_dofconnectivity(self, ndof: int):
        """Get the connectivity in terms of degrees of freedom

        Args:
            ndof: The number of degrees of freedom per node

        Returns:
            The dof numbers corresponding to each element of size (# total elements, # dofs per element)
        """
        return np.reshape(self.get_dofnumber(self.conn, ndof=ndof), (self.conn.shape[0], -1))

    def eval_shape_fun(self, pos: np.ndarray):
        r"""Evaluate the linear shape functions of the finite element

        In 1D
        .. math::
            N_1(x) = \frac{1}{w} \left(\frac{w}{2} - x\right)

            N_2(x) = \frac{1}{w} \left(\frac{w}{2} + x\right)

        In 2D [1]
        .. math::
            N_1(x,y) = \frac{1}{A} \left(\frac{w}{2} - x\right) \left(\frac{h}{2} - y\right)

            N_2(x,y) = \frac{1}{A} \left(\frac{w}{2} + x\right) \left(\frac{h}{2} - y\right)

            N_3(x,y) = \frac{1}{A} \left(\frac{w}{2} - x\right) \left(\frac{h}{2} + y\right)

            N_4(x,y) = \frac{1}{A} \left(\frac{w}{2} + x\right) \left(\frac{h}{2} + y\right)

        with :math:`A = wh`

        In 3D
        .. math::
            N_1(x,y,z) = \frac{1}{V} \left(\frac{w}{2}-x\right) \left(\frac{h}{2}-y\right) \left(\frac{d}{2}-z\right)

            \dotsc

        with :math:`V = whd`

        Args:
            pos : Evaluation coordinates [x, y, z (optional)] within bounds of [-element_size/2, element_size/2]

        Returns:
            Array of evaluated shape functions [N1(x), N2(x), ...]

        References:
            [1] Cook, et al. (2002). Concepts and applications of finite element analysis (4th ed.), eq. (6.2-3)
        """
        v = np.prod(self.element_size[: self.dim])
        assert v > 0.0, "Element volume needs to be positive"
        shapefn = np.ones(self.elemnodes) / v
        for i in range(self.dim):
            shapefn *= np.array([self.element_size[i] / 2 + n[i] * pos[i] for n in self.node_numbering])
        return shapefn

    def eval_shape_fun_der(self, pos: np.ndarray):
        """Evaluates the shape function derivatives in x, y, and optionally z-direction.
        For 1D domains, the y and z directions are optional.
        For 2D domains, the z direction is optional.

        Args:
            pos : Evaluation coordinates [x, y, z(optional)] within bounds of [-element_size/2, element_size/2]

        Returns:
            Shape function derivatives of size (#dimensions, #shape functions)
        """
        v = np.prod(self.element_size[: self.dim])
        assert v > 0.0, "Element volume needs to be positive"
        dN_dx = np.ones((self.dim, self.elemnodes)) / v  # dN/dx = 1/V
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:  # dN/dx_i *= (w[j]/2 ± x[j])
                    dN_dx[i, :] *= np.array([self.element_size[j] / 2 + n[j] * pos[j] for n in self.node_numbering])
            dN_dx[i, :] *= np.array([n[i] for n in self.node_numbering])  # Flip +/- signs according to node position
        return dN_dx

    def plot(self, ax, deformation=None, scaling=None):
        patches = []
        for e in range(self.nel):
            n = self.conn[e]
            x, y = self.get_node_position(n)
            u, v = deformation[n * 2], deformation[n * 2 + 1]
            color = (1 - scaling[e], 1 - scaling[e], 1 - scaling[e]) if scaling is not None else "grey"
            patch = plot_deformed_element(ax, x + u, v + y, linewidth=0.1, color=color)
            patches.append(patch)
        return patches

    def update_plot(self, patches, deformation=None, scaling=None):
        for e in range(self.nel):
            patch = patches[e]
            n = self.conn[e]
            x, y = self.get_node_position(n)
            u, v = deformation[n * 2], deformation[n * 2 + 1]
            color = (1 - scaling[e], 1 - scaling[e], 1 - scaling[e]) if scaling is not None else "grey"
            patch.set_color(color)
            patch.set_path(self.get_path(x + u, y + v))

    # flake8: noqa: C901
    def write_to_vti(self, vectors: dict, filename="out.vti", scale=1.0):
        """Write all given vectors to a Paraview (VTI) file

        The size of the vectors should be a multiple of ``nel`` or ``nnodes``. Based on their size they are marked as
        cell-data or point-data in the VTI file. For 2D data (size is equal to ``2*nnodes``), the z-dimension is padded
        with zeros to have 3-dimensional data. Also block-vectors of multiple dimensions (*e.g.* ``(2, 3*nnodes)``) are
        accepted, which get the suffixed as ``_00``.

        Args:
            vectors: A dictionary of vectors to write. Keys are used as vector names.
            filename (str): The file loction
            scale: Uniform scaling of the gridpoints
        """
        ext = ".vti"
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

        len_enc = ("<" if sys.byteorder == "little" else ">") + "Q"

        with open(filename, "wb") as file:
            # XML header
            file.write(b'<?xml version="1.0"?>\n')

            # Vtk header
            byte_order = "LittleEndian" if sys.byteorder == "little" else "BigEndian"
            file.write(
                f'<VTKFile type="ImageData" version="0.1" header_type="UInt64" byte_order="{byte_order}">\n'.encode()
            )

            # Extend of coordinates
            file.write(f'<ImageData WholeExtent="0 {self.nelx} 0 {self.nely} 0 {self.nelz}"'.encode())

            # Origin of domain
            file.write(f' Origin="{self.origin[0] * scale} {self.origin[1] * scale} {self.origin[2] * scale}"'.encode())

            # Spacing of points (dx, dy, dz)
            dx, dy, dz = self.element_size[0:3] * scale
            file.write(f' Spacing="{dx} {dy} {dz}">\n'.encode())

            # Start new piece
            file.write(f'<Piece Extent="0 {self.nelx} 0 {self.nely} 0 {self.nelz}">\n'.encode())

            # Start writing pointdata
            if len(point_dat) > 0:
                file.write(b"<PointData>\n")
                for key, vec in point_dat.items():
                    vecax = next((i for i, s in enumerate(vec.shape) if s % self.nnodes == 0), None)
                    ncomponents = vec.shape[vecax] // self.nnodes
                    # Convert 2D vectors to 3D, by padding with 0's (this enables the deform button in Paraview)
                    pad_to_vector = ncomponents == 2 and self.dim == 2
                    assert vec.ndim <= 2, "Only for 1D and 2D numpy arrays"
                    nvectors = 1 if vec.ndim == 1 else vec.shape[(vecax + 1) % 2]
                    nzeros = int(np.ceil(np.log10(nvectors)))
                    for i in range(nvectors):
                        vecname = key
                        if nvectors > 1:
                            intstr = i.__format__(f"0{nzeros}d")
                            vecname += f"({intstr})"
                            ind = [slice(None), slice(None)]
                            ind[(vecax + 1) % 2] = i
                            veci = vec[tuple(ind)]
                        else:
                            veci = vec

                        if np.iscomplexobj(veci):
                            vecs_to_write = [veci.real.astype(np.float32), veci.imag.astype(np.float32)]
                            vecs_name = [vecname + "(real)", vecname + "(imag)"]
                        else:
                            vecs_to_write = [veci.astype(np.float32)]
                            vecs_name = [vecname]

                        for v, t in zip(vecs_to_write, vecs_name):
                            if pad_to_vector:
                                vec_pad = np.zeros(3 * self.nnodes, dtype=np.float32)
                                vec_pad[0::3] = v[0::2]
                                vec_pad[1::3] = v[1::2]
                                v = vec_pad

                            file.write(
                                f'<DataArray type="Float32" '
                                f'Name="{t}" '
                                f'NumberOfComponents="{3 if pad_to_vector else ncomponents}" '
                                f'format="binary">\n'.encode()
                            )
                            enc_data = base64.b64encode(v)  # Encode the data
                            # Get the length of encoded data block
                            enc_len = base64.b64encode(struct.pack(len_enc, len(enc_data)))
                            file.write(enc_len)  # Write length
                            file.write(enc_data)  # Write data
                            file.write(b"\n</DataArray>\n")
                file.write(b"</PointData>\n")

            # Start writing celldata
            if len(cell_dat) > 0:
                file.write(b"<CellData>\n")
                for key, vec in cell_dat.items():
                    vecax = next((i for i, s in enumerate(vec.shape) if s % self.nel == 0), None)
                    ncomponents = vec.shape[vecax] // self.nel
                    assert vec.ndim <= 2, "Only for 1D and 2D numpy arrays"
                    nvectors = 1 if vec.ndim == 1 else vec.shape[(vecax + 1) % 2]

                    for i in range(nvectors):
                        vecname = key
                        if nvectors > 1:
                            vecname += f"({i})"
                            ind = [slice(None), slice(None)]
                            ind[(vecax + 1) % 2] = i
                            veci = vec[tuple(ind)]
                        else:
                            veci = vec

                        if np.iscomplexobj(veci):
                            vecs_to_write = [veci.real.astype(np.float32), veci.imag.astype(np.float32)]
                            vecs_name = [vecname + "(real)", vecname + "(imag)"]
                        else:
                            vecs_to_write = [veci.astype(np.float32)]
                            vecs_name = [vecname]

                        for v, t in zip(vecs_to_write, vecs_name):
                            file.write(
                                f'<DataArray type="Float32" '
                                f'Name="{t}" '
                                f'NumberOfComponents="{ncomponents}" '
                                f'format="binary">\n'.encode()
                            )
                            enc_data = base64.b64encode(v)  # Encode the data
                            # Get the length of encoded data block
                            enc_len = base64.b64encode(struct.pack(len_enc, len(enc_data)))
                            file.write(enc_len)  # Write length
                            file.write(enc_data)  # Write data
                            file.write(b"\n</DataArray>\n")
                file.write(b"</CellData>\n")

            file.write(b"</Piece>\n")
            file.write(b"</ImageData>\n")
            file.write(b"</VTKFile>")

    @staticmethod
    def create_for_mesh(mesh: Union[MeshT, list[MeshT]], h: float, Nmin: int = 1, Npadding: int = 0):
        """ Make a suitable domain for given (triangle) meshes (see :py:module:`numpy-stl`)

        Args:
            meshes: List of mesh objects that should fit inside the domain
            h: The element size
            Nmin (optional): Minimum common divisor in the shape (#elements) of the domain
            Npadding (optional): Minimum number of elements to add as padding on all sides
        """
        meshes = _parse_to_list(mesh)

        # Determine mesh extents
        mmin = np.minimum.reduce([m.min_ for m in meshes])
        mmax = np.maximum.reduce([m.max_ for m in meshes])

        # Determine number of elements
        n_elem = np.ceil(((mmax - mmin)/h + 2*Npadding)/Nmin).astype(int) * Nmin

        assert np.all(n_elem % Nmin == 0)
        assert np.all((n_elem - 2*Npadding) * h >= (mmax - mmin))

        # Determine domain extents
        dmin = (mmax + mmin - n_elem * h)/2  # == origin
        dmax = dmin + n_elem * h
        assert np.all(((dmin + dmax) / 2) == ((mmax + mmin) / 2))  # midpoints must align

        fmt = '% .2f'
        print(f"Mesh extents:   x = {fmt%mmin[0]} ... {fmt%mmax[0]}; y = {fmt%mmin[1]} ... {fmt%mmax[1]}; z = {fmt%mmin[2]} ... {fmt%mmax[2]}")
        print(f"Domain extents: x = {fmt%dmin[0]} ... {fmt%dmax[0]}; y = {fmt%dmin[1]} ... {fmt%dmax[1]}; z = {fmt%dmin[2]} ... {fmt%dmax[2]}")
        print(f"Domain size: {n_elem}")

        # Create domain
        domain = VoxelDomain(n_elem[0], n_elem[1], n_elem[2], unitx=h, unity=h, unitz=h)
        domain.origin = dmin

        return domain
    
    def voxelize(self, mesh: MeshT):
        
        dmin = self.origin
        dmax = self.origin + self.element_size * self.size

        # Check if mesh extents fit in domain
        if not np.all(mesh.min_ >= dmin) or not np.all(mesh.max_ <= dmax):
            warnings.warn(f"Mesh {mesh.name} does not fit in domain!")

        n_hits = np.zeros(self.size, dtype=int)
        for i0 in range(3):
            i1 = (i0 + 1) % 3
            i2 = (i0 + 2) % 3

            # Do ray-trace
            # Based on Moller-Trumbore algorithm https://doi.org/10.1080/10867651.1997.10487468
            direc = np.zeros(3)
            direc[i0] = 1.0

            v0v1 = mesh.v1 - mesh.v0
            v0v2 = mesh.v2 - mesh.v0
            pvec = np.cross(direc, v0v2)
            det = np.sum(v0v1 * pvec, axis=1)

            # Don't calculate for parallel facets
            parallel = np.abs(det)**2 < (1e-8**2) * np.sum(v0v1 * v0v1, axis=1) * np.sum(v0v2 * v0v2, axis=1)

            aligned = np.ones_like(det)
            aligned[det > 0] = -1.0

            # Calculate ray origins
            t1_range = self.element_size[i1]/2 + np.linspace(dmin[i1], dmax[i1], self.size[i1]+1)[:-1]
            t2_range = self.element_size[i2]/2 + np.linspace(dmin[i2], dmax[i2], self.size[i2]+1)[:-1]

            ray_origin = np.zeros((t1_range.size, t2_range.size, 3))
            ray_origin[..., i0] = dmin[i0]
            ray_origin[..., i1] = t1_range[:, None]
            ray_origin[..., i2] = t2_range[None, :]

            # Find ray intersections by calculating barycentric coordinates on the triangles
            # Shape is (#facets, #t1, #t2, #dim)
            # First coordinate
            tvec = ray_origin[None, ...] - mesh.v0[~parallel, None, None, :]
            uvw = np.zeros_like(tvec)
            uvw[..., 1] = np.sum(tvec * pvec[~parallel, None, None, :], axis=-1) / det[~parallel, None, None]

            # Second coordinate
            qvec = np.cross(tvec, v0v1[~parallel, None, None, :])
            uvw[..., 2] = (qvec @ direc) / det[~parallel, None, None]

            # Third coordinate
            uvw[..., 0] = 1.0 - uvw[..., 1] - uvw[..., 2]

            # Distance from origin to facet
            t = np.sum(v0v2[~parallel, None, None, :] * qvec, axis=-1) / det[~parallel, None, None]

            # Find intersected instances
            intersected = np.logical_and(np.all(uvw >= 0, axis=-1), np.all(uvw <= 1, axis=-1))

            n_intersections = np.sum(intersected, axis=0)  # Number of intersections per ray
            if not np.all((n_intersections % 2) == 0):
                warnings.warn("Inersections found with unequal number of entries and exits")

            # Sort hits based on distance
            distances = t[intersected]
            indices = np.argwhere(intersected)

            order = np.lexsort((distances, indices[:, 1], indices[:, 2]))
            aligned[~parallel][indices[order, 0]]

            idx_facet = indices[order, 0]
            idx_ray_t1 = indices[order, 1]
            idx_ray_t2 = indices[order, 2]
            hit_dist = distances[order]
            hit_direc = aligned[~parallel][idx_facet]

            # np.hstack([indices[order], distances[order, None], aligned[~parallel][indices[order, 0], None]])

            # Find lines from hit entry to hit exit
            i_lines = np.argwhere((np.diff(hit_direc) == 2) *
                                (np.diff(idx_ray_t1) == 0) *
                                (np.diff(idx_ray_t2) == 0)).flatten()

            assert np.all(idx_ray_t1[i_lines] == idx_ray_t1[i_lines + 1])
            assert np.all(idx_ray_t2[i_lines] == idx_ray_t2[i_lines + 1])
            assert np.all(hit_direc[i_lines] == -1)
            assert np.all(hit_direc[i_lines+1] == 1)
            assert np.all(hit_dist[i_lines] <= hit_dist[i_lines+1])

            idx0_line_start = np.floor(hit_dist[i_lines] / self.element_size[i0]).astype(int)
            idx0_line_end = np.ceil(hit_dist[i_lines+1] / self.element_size[i0]).astype(int)
            idx1_line = idx_ray_t1[i_lines]
            idx2_line = idx_ray_t2[i_lines]

            # if intersected.sum() / 2 > i_lines.size:
            #     warnings.warn(f"Found {intersected.sum()} intersections but only {i_lines.size} lines")

            is_hit = np.zeros(self.size, dtype=int)
            if i0 == 0:
                idx_line = np.arange(self.size[i0])[:, None]
                idx_intersect = (idx_line >= idx0_line_start[None, :]) * (idx_line <= idx0_line_end[None, :])
                np.add.at(is_hit, (slice(None), idx1_line, idx2_line), idx_intersect)
            elif i0 == 1:
                idx_line = np.arange(self.size[i0])[None, :]
                idx_intersect = (idx_line >= idx0_line_start[:, None]) * (idx_line <= idx0_line_end[:, None])
                np.add.at(is_hit, (idx2_line, slice(None), idx1_line), idx_intersect)
            elif i0 == 2:
                idx_line = np.arange(self.size[i0])[None, :]
                idx_intersect = (idx_line >= idx0_line_start[:, None]) * (idx_line <= idx0_line_end[:, None])
                np.add.at(is_hit, (idx1_line, idx2_line, slice(None)), idx_intersect)
            n_hits += np.clip(is_hit, 0, 1)

        return self.elements[n_hits >= 2]