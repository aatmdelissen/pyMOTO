from pymoto import Module, DomainDefinition
import numpy as np
from scipy.sparse import coo_matrix
from scipy.signal import convolve, correlate
from numbers import Number


class FilterConv(Module):
    r"""Density filter based on convolution

    :math:`y_{i,j,k} = W \ast x_{i,j,k} = \sum_{p,q,r} W_{p,q,r} x_{i-p,j-q,k-r}`

    Input Signal:
        - ``x``: The unfiltered field :math:`\mathbf{x}`

    Output Signal:
        - ``y``: Filtered field :math:`\mathbf{y}`

    References:
        [Wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing))
    """

    def __init__(
        self,
        domain: DomainDefinition,
        radius: float = None,
        relative_units: bool = True,
        weights: np.ndarray = None,
        xmin_bc="symmetric",
        xmax_bc="symmetric",
        ymin_bc="symmetric",
        ymax_bc="symmetric",
        zmin_bc="symmetric",
        zmax_bc="symmetric",
    ):
        r"""Initialize density filter module based on convolution

        Either the argument filter radius (`radius`) or a filtering kernel `weights` needs to be provided. If a filter
        radius is passed, the standard linear density filter will be used (see :py:class:`pymoto.DensityFilter`).

        For the boundaries, a padded effect can be selected from the following options:

        'symmetric' (default)
            Pads with the reflection of the vector mirrored
            along the edge of the array.

        value (e.g. `1.0` or `0.0`)
            Pads with a constant value.

        'edge'
            Pads with the edge values of array.

        'wrap'
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The (finite-element) domain
            radius (float, optional): Filter radius. If this is not provided, the filtering kernel `weights` must be
              defined
            relative_units (bool, optional): Indicate if the filter radius is in relative units with respect to the
              element-size or is given as an absolute geometry using element size. Defaults to True.
            weights (np.ndarray, optional): Use a custom filtering kernel (2D or 3D array). Alternatively, the filter
              `radius` can be provided to have the kernel initialized automatically.
            xmin_bc (str, optional): Boundary condition for the boundary at minimum x-value. Defaults to "symmetric".
            xmax_bc (str, optional): Boundary condition for the boundary at maximum x-value. Defaults to "symmetric".
            ymin_bc (str, optional): Boundary condition at minimum y. Defaults to "symmetric".
            ymax_bc (str, optional): Boundary condition at maximum y. Defaults to "symmetric".
            zmin_bc (str, optional): Boundary condition at minimum z (only in 3D). Defaults to "symmetric".
            zmax_bc (str, optional): Bounadry condition at maximum z (only in 3D). Defaults to "symmetric".
        """
        self.domain = domain
        self.weights = None
        if (weights is None and radius is None) or (weights is not None and radius is not None):
            raise ValueError("Only one of arguments 'filter_radius' or 'weights' must be provided.")
        elif weights is not None:
            self.weights = weights.copy()
            while self.weights.ndim < 3:
                self.weights = np.expand_dims(self.weights, axis=-1)
            for i in range(self.weights.ndim):
                assert self.weights.shape[i] % 2 == 1, "Size of weights must be uneven"
        elif radius is not None:
            self.set_filter_radius(radius, relative_units)

        # Process padding
        self.overrides = []
        self.pad_sizes = [v // 2 for v in self.weights.shape]

        domain_sizes = [self.domain.nelx, self.domain.nely, self.domain.nelz]
        el_x, el_y, el_z = np.meshgrid(*[np.arange(max(1, s)) for s in domain_sizes], indexing="ij")
        self.el3d_orig = self.domain.get_elemnumber(el_x, el_y, el_z)

        padx = self._process_padding(self.el3d_orig, xmin_bc, xmax_bc, 0, self.pad_sizes[0])
        pady = self._process_padding(padx, ymin_bc, ymax_bc, 1, self.pad_sizes[1])
        self.el3d_pad = self._process_padding(pady, zmin_bc, zmax_bc, 2, self.pad_sizes[2])

    def _process_padding(self, indices, type_edge0, type_edge1, direction: int, pad_size: int):
        # First process wrapped padding
        wrap_size = (0, 0)
        do_wrap = False
        if type_edge0 == "wrap":
            do_wrap = True
            wrap_size = (pad_size, wrap_size[1])
        if type_edge1 == "wrap":
            do_wrap = True
            wrap_size = (wrap_size[0], pad_size)

        if do_wrap:
            pad_width = [(0, 0) for _ in range(indices.ndim)]
            pad_width[direction] = wrap_size
            pad1a = np.pad(indices, pad_width, mode="wrap")
        else:
            pad1a = indices

        domain_sizes = [self.domain.nelx, self.domain.nely, self.domain.nelz]
        padded_sizes = [
            self.domain.nelx + 2 * self.pad_sizes[0],
            self.domain.nely + 2 * self.pad_sizes[1],
            self.domain.nelz + 2 * self.pad_sizes[2],
        ]

        # Process edge 1
        pad_width = [(0, 0) for _ in range(indices.ndim)]
        pad_width[direction] = (0, pad_size)
        if type_edge1 == "edge":
            pad1b = np.pad(pad1a, pad_width, mode="edge")
        elif type_edge1 == "symmetric":
            pad1b = np.pad(pad1a, pad_width, mode="symmetric")
        elif isinstance(type_edge1, Number):  # Constant
            value = type_edge1
            pad1b = np.pad(pad1a, pad_width, mode="constant", constant_values=0)

            n_range = [np.arange(max(1, s)) for s in padded_sizes]
            n_range[direction] = pad_size + domain_sizes[direction] + np.arange(pad_size)

            el_nos = np.meshgrid(*n_range, indexing="ij")
            self.override_padded_values(tuple(el_nos), value)
        else:
            pad1b = pad1a

        # Process edge 0
        pad_width = [(0, 0) for _ in range(indices.ndim)]
        pad_width[direction] = (pad_size, 0)
        if type_edge0 == "edge":
            pad1 = np.pad(pad1b, pad_width, mode="edge")
        elif type_edge0 == "symmetric":
            pad1 = np.pad(pad1b, pad_width, mode="symmetric")
        elif isinstance(type_edge0, Number):
            value = type_edge0
            pad1 = np.pad(pad1b, pad_width, mode="constant", constant_values=0)

            n_range = [np.arange(max(1, s)) for s in padded_sizes]
            n_range[direction] = np.arange(pad_size)

            el_nos = np.meshgrid(*n_range, indexing="ij")
            self.override_padded_values(tuple(el_nos), value)
        else:
            pad1 = pad1b
        return pad1

    @property
    def padded_domain(self):
        domain_sizes = [self.domain.nelx, self.domain.nely, self.domain.nelz]
        nx, ny, nz = [n + 2 * p for n, p in zip(domain_sizes, self.pad_sizes)]
        lx, ly, lz = self.domain.element_size
        return DomainDefinition(nx, ny, nz, unitx=lx, unity=ly, unitz=lz)

    def override_padded_values(self, index, value):
        if all([np.asarray(i).size == 0 for i in index]):
            # Don't add empty sets
            return
        self.overrides.append((index, value))

    def override_values(self, index, value):
        # Change index to extended domain
        xrange = self.pad_sizes[0] + np.arange(max(1, self.domain.nelx))
        yrange = self.pad_sizes[1] + np.arange(max(1, self.domain.nely))
        zrange = self.pad_sizes[2] + np.arange(max(1, self.domain.nelz))
        el_x, el_y, el_z = np.meshgrid(xrange, yrange, zrange, indexing="ij")
        self.overrides.append(((el_x[index], el_y[index], el_z[index]), value))

    def get_padded_vector(self, x):
        xpad = x[self.el3d_pad]
        for index, value in self.overrides:
            xpad[index] = value
        return xpad

    def set_filter_radius(self, radius: float, relative_units: bool = True):
        if relative_units:
            dx, dy, dz = 1.0, 1.0, 1.0
        else:
            dx, dy, dz = self.domain.element_size
        nx, ny, nz = self.domain.nelx, self.domain.nely, self.domain.nelz
        delemx = min(nx, int((radius - 1e-10 * dx) / dx))
        delemy = min(ny, int((radius - 1e-10 * dy) / dy))
        delemz = min(nz, int((radius - 1e-10 * dz) / dz))
        xrange = np.arange(-delemx, delemx + 1) * dx
        yrange = np.arange(-delemy, delemy + 1) * dy
        zrange = np.arange(-delemz, delemz + 1) * dz
        coords_x, coords_y, coords_z = np.meshgrid(xrange, yrange, zrange, indexing="ij")
        self.weights = np.maximum(
            0.0, radius - np.sqrt(coords_x * coords_x + coords_y * coords_y + coords_z * coords_z)
        )
        self.weights /= np.sum(self.weights)  # Volume preserving

    def __call__(self, x):
        xpad = self.get_padded_vector(x)
        y3d = convolve(xpad, self.weights, mode="valid")
        y = np.zeros_like(x)
        np.add.at(y, self.el3d_orig, y3d)
        return y

    def _sensitivity(self, dfdv):
        dx3d = correlate(dfdv[self.el3d_orig], self.weights, mode="full")
        for index, _ in self.overrides:
            dx3d[index] = 0
        dx = np.zeros_like(self.sig_in[0].state)
        np.add.at(dx, self.el3d_pad, dx3d)
        return dx


class Filter(Module):
    r"""Abstract base class for any linear filter with normalization

    This module carries out the mathematical operation
    :math:`\mathbf{y} = \mathbf{S}^{-1} \mathbf{H}\mathbf{x}` in which :math:`\mathbf{S}=\text{diag}(\mathbf{s})` is a
    diagonal matrix. In index notation the same relation is written as
    :math:`y_i = \frac{\sum_j H_{ij} x_j}{ s_i }`.

    The normalization vector is the row-wise sums of :math:`\mathbf{H}`, of which the entries are calculated as
    :math:`s_i = \sum_j H_{ij}`.

    Input Signal:
        - ``x``: The unfiltered field :math:`\mathbf{x}`

    Output Signal:
        - ``y``: Filtered field :math:`\mathbf{y}`
    """

    def __init__(self, *args, nonpadding=None, **kwargs):
        r"""Initialize abstract base-class for linear filters

        Args:
            nonpadding (numpy.array[int], optional): An array with indices at places where
              :math:`s_i = \max(\mathbf{s}) \: \forall\: i \notin \mathcal{N}`. For a density filter this mimics having
              values of `0` outside of the domain, thus emulating padding of the boundaries.
        """
        self.H = self._calculate_h(*args, **kwargs).tocsc()

        self.Hs = self.H.sum(1)

        if nonpadding is not None:
            inds = ~np.isin(np.arange(len(self.Hs)), nonpadding)
            self.Hs[inds] = np.max(self.Hs)

    @staticmethod
    def _calculate_h(*args, **kwargs):
        r"""This method should be overridden by any child-classes to implement their own filtering behavior

        Returns:
            Filtering matrix :math:`\mathbf{H}`, *e.g.* in COO-format
        """
        raise NotImplementedError("Filter not implemented.")

    def __call__(self, x):
        return np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

    def _sensitivity(self, dfdy):
        return np.asarray(self.H * (dfdy[np.newaxis].T / self.Hs))[:, 0]


class DensityFilter(Filter):
    r"""Standard density filter for a structured mesh in topology optimization

    The filtered densities are calculated as

    :math:`y_i = \sum_j \frac{H_{ij}}{s_i}x_j`,

    where :math:`H_{ij}=\max \left( r - \sqrt{ (x_j - x_i)^2 + (y_j - y_i)^2 + (z_j - z_i)^2 } , 0 \right)`,

    and :math:`s_i=\sum_j H_{ij}`.

    Input Signal:
        - ``x``: The unfiltered field :math:`\mathbf{x}`

    Output Signal:
        - ``y``: Filtered field :math:`\mathbf{y}`

    Args:
        domain (:py:class:`pymoto.DomainDefinition`): The finite element domain

    Keyword Args:
        radius (float or int): The filtering radius (in absolute units of elements)
        nonpadding (numpy.array[int]): An array with indices at places where
          :math:`s_i = \max(\mathbf{s}) \: \forall\: i \notin \mathcal{N}`. For a density filter this mimics having
          values of ``0`` outside of the domain, thus emulating padding of the boundaries.

    References:
      - Bruns & Tortorelli (2001). *Topology optimization of non-linear elastic structures and compliant mechanisms*.
        Computer Methods in Applied Mechanics and Engineering, 190(26–27), 3443–3459.
        `doi: 10.1016/S0045-7825(00)00278-4 <https://doi.org/10.1016/S0045-7825(00)00278-4>`_
      - Bourdain (2001). *Filters in topology optimization*. International Journal for Numerical Methods in
        Engineering, 50, 2143-2158. `doi: 10.1002/nme.116 <https://doi.org/10.1002/nme.116>`_
    """

    @staticmethod
    def _calculate_h(domain: DomainDefinition, radius=2.0):
        """Density filter: Build (and assemble) the index+data vectors for the coo matrix format
        Total number of filter entries - for every element, a number of entries wrt other elements are needed (not
        including boundaries)
        Takes a square domain around current element, encompassing circle with radius floor(rmin)
        """
        delem = int(radius)
        # Number of elements
        nx, ny, nz = domain.nelx, domain.nely, max(domain.nelz, 1)
        nel = domain.nel  # Number of elements

        xrange = np.arange(0, nx)
        yrange = np.arange(0, ny)
        zrange = np.arange(0, nz)

        # Get element grid with indices in x, y, and z directions
        xinds, yinds, zinds = np.meshgrid(xrange, yrange, zrange, indexing="ij")

        # Obtain global element numbers
        els = domain.get_elemnumber(xinds, yinds, zinds)

        # Rearrange directional indices according to element number
        ix = np.zeros(nel, dtype=int)
        iy = np.zeros(nel, dtype=int)
        iz = np.zeros(nel, dtype=int)
        ix[els] = xinds
        iy[els] = yinds
        iz[els] = zinds

        # Determine the limits of the window wrt current element
        xlow = np.maximum(ix - delem, 0)
        xupp = np.minimum(ix + delem, nx - 1)
        ylow = np.maximum(iy - delem, 0)
        yupp = np.minimum(iy + delem, ny - 1)
        zlow = np.maximum(iz - delem, 0)
        zupp = np.minimum(iz + delem, nz - 1)

        # Number of window elements in x, y and z direction and the total number per element
        nwindx = xupp - xlow + 1
        nwindy = yupp - ylow + 1
        nwindz = zupp - zlow + 1
        nwind = nwindx * nwindy * nwindz
        if np.sum(nwind) < 0:
            raise OverflowError("Filter size too large for this mesh size")

        # Total number of window elements
        ncum = np.cumsum(nwind)

        nfilter = ncum[-1]

        # Initialize sparse indices for coo format
        h_rows = np.zeros(nfilter, dtype=np.uint32)
        h_cols = np.zeros(nfilter, dtype=np.uint32)

        # For all elements, get the surrounding elements
        for el in range(nel):
            elcomp = els[xlow[el] : xupp[el] + 1, ylow[el] : yupp[el] + 1, zlow[el] : zupp[el] + 1].flatten()

            indstart = ncum[el - 1] if el > 0 else 0
            h_rows[indstart : ncum[el]] = el
            h_cols[indstart : ncum[el]] = elcomp

        # Calculate element distances
        dx = ix[h_rows] - ix[h_cols]
        dy = iy[h_rows] - iy[h_cols]
        dz = iz[h_rows] - iz[h_cols]

        h_values = np.maximum(0.0, radius - np.sqrt(dx * dx + dy * dy + dz * dz))

        # Finalize assembly
        return coo_matrix((h_values, (h_rows, h_cols)), shape=(nel, nel))


class OverhangFilter(Module):
    r"""Implementation of overhang filter by Langelaar (2016, 2017)

    It proceeds layer by layer through the entire domain. For each element in the current layer, the maximum printable
    density is determined by a smooth maximum of the supporting elements
    :math:`s_i = \text{smax}(\mathbf{y}_\text{supp})`. Then, the final printed density value is obtained by a smooth
    minimum operation of the desired density
    :math:`x_i` and the maximum printable density :math:`x_i` as :math:`y_i = \text{smin}(x_i, s_i)`.

    Input Signal:
        - ``x``: The unfiltered field :math:`\mathbf{x}`

    Output Signal:
        - ``y``: Filtered field :math:`\mathbf{y}`, without overhangs

    References:
      - Langelaar, M. (2017). *An additive manufacturing filter for topology optimization of print-ready designs*.
        Structural and Multidisciplinary Optimization, 55(3), 871-883.
        `doi: 10.1007/s00158-016-1522-2 <https://doi.org/10.1007/s00158-016-1522-2>`_
      - Langelaar, M. (2016). *Topology optimization of 3D self-supporting structures for additive manufacturing*.
        Additive Manufacturing, 12, 60-70.
        `doi: 10.1016/j.addma.2016.06.010 <https://doi.org/10.1016/j.addma.2016.06.010>`_
    """

    def __init__(
        self,
        domain: DomainDefinition,
        direction=(0.0, 1.0, 0.0),
        xi_0: float = 0.5,
        p: float = 40.0,
        eps: float = 1e-4,
        nsampling: int = None,
    ):
        """Initialize overhang filter modulue

        Args:
            domain (:py:class:`pymoto.DomainDefinition`): The (finite-element) domain
            direction (tuple, optional): Print direction as array or string, e.g. ``[0, -1]`` (in 2D) or ``"y-"`` for
              negative y direction. Currently, only directions aligned with one of the Cartesian axes are supported.
              Default is ``[0, 1, 0]``
            xi_0 (float, optional): Density value for which zero overshoot is required ( ``0 <= xi_0 <= 1`` ).
              Default is ``0.5``
            p (float, optional): Exponent of the smooth maximum function ( ``p > 0`` ). Higher p increases accuracy, but
              reduces smoothness. Default is ``40.0``
            eps (float, optional): Smooth minimum regularization parameter ( ``eps >= 0`` ). Lower eps increases
              accuracy, but reduces smoothness. Default is ``1e-4``
            nsampling (int, optional): ``3`` for 2D overhang, ``5`` or ``9`` for 3D overhang. Default is ``3`` in 2D and
              ``5`` in 3D
        """
        # Set print direction
        self.direction = direction

        self.domain = domain
        if self.domain.dim == 2 and self.direction[2] != 0:
            raise ValueError("Z-direction must be zero for 2-dimensional domain")

        # Determine sampling pattern
        if nsampling is None:
            nsampling = 3 if self.domain.dim == 2 else 5
        if self.domain.dim == 2 and nsampling != 3:
            raise ValueError(f"For 2D domains, nsampling should be 3, not {nsampling}")
        if self.domain.dim == 3 and nsampling not in [5, 9]:
            raise ValueError(f"For 3D domains, nsampling should be 5 or 9, not {nsampling}")

        # Parameters
        self.xi_0 = xi_0
        self.p = p
        self.eps = eps
        self.nsampling = nsampling
        self.q, self.shift, self.backshift = None, None, None
        self.smax = None

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, val):
        # Parse print direction
        if isinstance(val, str):
            # Print axis
            axes = np.argwhere([a in val.lower() for a in ["x", "y", "z"]]).flatten()
            if axes.size != 1:
                raise ValueError(f'Wronly specified print direction {val}, should be e.g. "x", "+x", "-y"')

            # Print direction
            val = [0.0, 0.0, 0.0]
            val[axes[0]] = -1.0 if "-" in val else +1.0

        direction = np.asarray(val, dtype=float).ravel()
        if direction.size < 3:
            direction = np.pad(direction, (0, 3 - direction.size), "constant", constant_values=0.0)
        elif direction.size > 3:
            direction = direction[:3]

        if np.abs(direction).astype(int).sum() != 1:
            raise ValueError("The print direction must be aligned with either x, y or z directions")

        self._direction = direction / np.linalg.norm(direction)

    def set_parameters(self, typ: np.dtype):
        """Set the internal smooth-maximum and smooth-minimum parameters according to the values in Langelaar, 2017

        Args:
            typ: The numeric type used in the calculations
        """
        dbl_min = np.finfo(typ).tiny
        self.q = self.p + np.log(1.0 * self.nsampling) / np.log(self.xi_0)
        self.shift = 100.0 * pow(dbl_min, 1.0 / self.p)  # Small shift to prevent division by 0
        # Backshift 5% smaller to be on the safe side
        self.backshift = pow(self.nsampling, 1 / self.q) * pow(self.shift, self.p / self.q) * 0.95

    @property
    def layer_offsets(self):
        dir_layer = int(np.argmax(abs(self.direction)))  # The axis of the print direction
        dir_orth1 = (dir_layer + 1) % 3
        dir_orth2 = (dir_layer + 2) % 3
        if dir_orth1 == 2 and self.domain.dim == 2:
            dir_orth1, dir_orth2 = dir_orth2, dir_orth1  # Make sure the z-direction is last for 2D

        # Select layer offsets from:
        layer_offsets = np.zeros((9, 3), dtype=int)
        for i in range(3):  # 3-point 2D supports
            layer_offsets[i, dir_orth1] = i - 1
        for i in range(2):  # 5-point 3D supports
            layer_offsets[3 + i, dir_orth2] = -1 + 2 * i
        for i in range(2):  # 9-point 3D supports
            for j in range(2):
                layer_offsets[5 + 2 * i + j, dir_orth1] = -1 + 2 * j
                layer_offsets[5 + 2 * i + j, dir_orth2] = -1 + 2 * i
        layer_offsets = layer_offsets[: self.nsampling]
        return layer_offsets

    def __call__(self, x0):
        if self.q is None:  # Set parameters according to data type of x
            self.set_parameters(x0.dtype)

        # Make 3D array with densitities
        ei, ej, ek = np.indices((self.domain.nelx, self.domain.nely, max(self.domain.nelz, 1)))
        els = self.domain.get_elemnumber(ei, ej, ek)
        x = x0[els]

        # Make dataset for printable density and max supported density
        xprint = np.zeros_like(x)
        self.smax = np.zeros_like(x)

        # Indices
        dir_layer = int(np.argmax(abs(self.direction)))  # The axis of the print direction
        dx_layer = int(np.sign(self.direction[dir_layer]))  # Iteration direction
        ind0_layer = 1 if dx_layer >= 0 else x.shape[dir_layer] - 2  # Starting index

        # Local support indices
        layer_offsets = self.layer_offsets

        # Support layer indices
        support_shape = np.asarray(x.shape)
        support_shape[dir_layer] = 1
        sel_layer = list(np.indices(support_shape, sparse=True))

        # Determine padding size
        pad_size = np.zeros((3, 2), dtype=int)
        pad_size[:, 0] = -layer_offsets.min(axis=0)  # Padding on negative side
        pad_size[:, 1] = layer_offsets.max(axis=0)  # Padding on positive side
        origin = pad_size[:, 0]
        pad_size = tuple(tuple(p) for p in pad_size)

        # Initial layer is on the baseplate, so is always printable
        sel_layer[dir_layer][...] = ind0_layer - dx_layer
        xprint[tuple(sel_layer)] = x[tuple(sel_layer)]

        # Loop over all the other layers
        for i in range(x.shape[dir_layer] - 1):
            # 1) Get support layer
            sel_layer[dir_layer][...] = ind0_layer - dx_layer
            xsupp = np.pad(xprint[tuple(sel_layer)], pad_size, constant_values=0)

            xsum = np.zeros_like(xprint[tuple(sel_layer)])
            for offset in layer_offsets:
                el_supp = [origin[i] + offset[i] + sel_layer[i] for i in range(3)]
                el_supp[dir_layer][...] = 0
                xsum += np.power(xsupp[tuple(el_supp)] + self.shift, self.p)

            # 2) Take smooth maximum
            max_supp = np.power(xsum, 1 / self.q) - self.backshift
            # max_supp = np.maximum.reduce(supp_vals)  # Absolute maximum

            # 3) Take smooth minimum
            sel_layer[dir_layer][...] = ind0_layer
            self.smax[tuple(sel_layer)] = max_supp  # Save maximum printable densities for sensitivity
            r1 = x[tuple(sel_layer)] - max_supp
            xprint[tuple(sel_layer)] = (
                x[tuple(sel_layer)] + max_supp - np.sqrt(r1 * r1 + self.eps) + np.sqrt(self.eps)
            ) / 2
            # xprint[tuple(sel_layer)] = np.minimum(x[tuple(sel_layer)], max_supp)

            ind0_layer += dx_layer

        xprint_flat = np.zeros_like(x0)
        xprint_flat[els] = xprint
        return xprint_flat

    def _sensitivity(self, dxprint0):
        x0 = self.get_input_states()
        xprint0 = self.get_output_states()

        # Make 3D array with densitities
        ei, ej, ek = np.indices((self.domain.nelx, self.domain.nely, max(self.domain.nelz, 1)))
        els = self.domain.get_elemnumber(ei, ej, ek)
        x = x0[els]
        xprint = xprint0[els]
        dxprint = dxprint0[els]

        # Make dataset for sensitivties wrt input signal
        dx = np.zeros_like(dxprint)

        # Indices
        dir_layer = int(np.argmax(abs(self.direction)))  # The axis of the print direction
        dx_layer = int(np.sign(self.direction[dir_layer]))  # Iteration direction
        ind0_layer = x.shape[dir_layer] - 1 if dx_layer >= 0 else 0  # Starting index (="ending" in response)

        # Local support indices
        layer_offsets = self.layer_offsets

        # Support layer indices
        support_shape = np.asarray(x.shape)
        support_shape[dir_layer] = 1
        sel_layer = list(np.indices(support_shape, sparse=True))

        # Determine padding size
        pad_size = np.zeros((3, 2), dtype=int)
        pad_size[:, 0] = -layer_offsets.min(axis=0)  # Padding on negative side
        pad_size[:, 1] = layer_offsets.max(axis=0)  # Padding on positive side
        origin = pad_size[:, 0]
        pad_size = tuple(tuple(p) for p in pad_size)

        # Loop over all the layers
        for i in range(x.shape[dir_layer] - 1):
            # 3) Take smooth minimum
            sel_layer[dir_layer][...] = ind0_layer

            # FW: xprint[tuple(sel_layer)] = (x[tuple(sel_layer)] + max_supp - np.sqrt(r1*r1 + self.eps) +
            #                                 np.sqrt(self.eps))/2
            r1 = x[tuple(sel_layer)] - self.smax[tuple(sel_layer)]
            dfdr1 = -dxprint[tuple(sel_layer)] * r1 / (2 * np.sqrt(r1 * r1 + self.eps))
            dx[tuple(sel_layer)] += dxprint[tuple(sel_layer)] / 2 + dfdr1  # Direct contribution of x
            dfdsmax = dxprint[tuple(sel_layer)] / 2 - dfdr1  # Contribution through smax

            # 2) Take smooth maximum
            # FW: max_supp = np.power(keep, 1/self.q)-self.backshift
            xsum = np.power(self.smax[tuple(sel_layer)] + self.backshift, self.q)
            dfdxsum = dfdsmax * np.power(xsum, (1 / self.q) - 1) / self.q

            # 1) Get all support values
            sel_layer[dir_layer][...] = ind0_layer - dx_layer
            xsupp = np.pad(xprint[tuple(sel_layer)], pad_size, constant_values=0)
            dxsupp = np.zeros_like(xsupp)
            for offset in layer_offsets:
                el_supp = [origin[i] + offset[i] + sel_layer[i] for i in range(3)]
                el_supp[dir_layer][...] = 0
                # FW: xsum += np.power(xsupp[tuple(el_supp)] + self.shift, self.p)
                dxsupp[tuple(el_supp)] += self.p * dfdxsum * np.power(xsupp[tuple(el_supp)] + self.shift, self.p - 1)
            el_supp = [origin[i] + sel_layer[i] for i in range(3)]
            el_supp[dir_layer][...] = 0
            dxprint[tuple(sel_layer)] += dxsupp[tuple(el_supp)]

            ind0_layer -= dx_layer  # Traverse reverse direction

        # Base layer is directly transferred
        sel_layer[dir_layer][...] = ind0_layer  # dx_layer is already subtracted in the loop
        # FW: xprint[tuple(sel_layer)] = x[tuple(sel_layer)]
        dx[tuple(sel_layer)] += dxprint[tuple(sel_layer)]

        dx0 = np.zeros_like(dxprint0)
        dx0[els] = dx
        return dx0
