from pymoto import Module, DomainDefinition
import numpy as np
from scipy.sparse import coo_matrix
from scipy.signal import convolve, correlate
from numbers import Number


class FilterConv(Module):
    r""" Density filter based on convolution

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
        domain: The DomainDefinition
        radius (optional): Filter radius
        relative_units(optional): Indicate if the filter radius is in relative units with respect to the element-size or
          is given as an absolute size
        weights(optional): Filtering kernel (2D or 3D array)
        xmin_bc(optional): Boundary condition for the boundary at minimum x-value
        xmax_bc(optional): Boundary condition for the boundary at maximum x-value
        ymin_bc(optional): Boundary condition at minimum y
        ymax_bc(optional): Boundary condition at maximum y
        zmin_bc(optional): Boundary condition at minimum z (only in 3D)
        zmax_bc(optional): Bounadry condition at maximum z (only in 3D)
    """
    def _prepare(self, domain: DomainDefinition, radius: float = None, relative_units: bool = True, weights: np.ndarray = None,
                 xmin_bc='symmetric', xmax_bc='symmetric',
                 ymin_bc='symmetric', ymax_bc='symmetric',
                 zmin_bc='symmetric', zmax_bc='symmetric'):

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
        self.pad_sizes = [v//2 for v in self.weights.shape]

        domain_sizes = [self.domain.nelx, self.domain.nely, self.domain.nelz]
        el_x, el_y, el_z = np.meshgrid(*[np.arange(max(1, s)) for s in domain_sizes], indexing='ij')
        self.el3d_orig = self.domain.get_elemnumber(el_x, el_y, el_z)

        padx = self._process_padding(self.el3d_orig, xmin_bc, xmax_bc, 0, self.pad_sizes[0])
        pady = self._process_padding(padx, ymin_bc, ymax_bc, 1, self.pad_sizes[1])
        self.el3d_pad = self._process_padding(pady, zmin_bc, zmax_bc, 2, self.pad_sizes[2])

    def _process_padding(self, indices, type_edge0, type_edge1, direction: int, pad_size: int):
        # First process wrapped padding
        wrap_size = (0, 0)
        do_wrap = False
        if type_edge0 == 'wrap':
            do_wrap = True
            wrap_size = (pad_size, wrap_size[1])
        if type_edge1 == 'wrap':
            do_wrap = True
            wrap_size = (wrap_size[0], pad_size)

        if do_wrap:
            pad_width = [(0, 0) for _ in range(indices.ndim)]
            pad_width[direction] = wrap_size
            pad1a = np.pad(indices, pad_width, mode='wrap')
        else:
            pad1a = indices

        domain_sizes = [self.domain.nelx, self.domain.nely, self.domain.nelz]
        padded_sizes = [self.domain.nelx + 2 * self.pad_sizes[0],
                        self.domain.nely + 2 * self.pad_sizes[1],
                        self.domain.nelz + 2 * self.pad_sizes[2]]

        # Process edge 1
        pad_width = [(0, 0) for _ in range(indices.ndim)]
        pad_width[direction] = (0, pad_size)
        if type_edge1 == 'edge':
            pad1b = np.pad(pad1a, pad_width, mode='edge')
        elif type_edge1 == 'symmetric':
            pad1b = np.pad(pad1a, pad_width, mode='symmetric')
        elif isinstance(type_edge1, Number):  # Constant
            value = type_edge1
            pad1b = np.pad(pad1a, pad_width, mode='constant', constant_values=0)

            n_range = [np.arange(max(1, s)) for s in padded_sizes]
            n_range[direction] = pad_size + domain_sizes[direction] + np.arange(pad_size)

            el_nos = np.meshgrid(*n_range, indexing='ij')
            self.override_padded_values(tuple(el_nos), value)
        else:
            pad1b = pad1a

        # Process edge 0
        pad_width = [(0, 0) for _ in range(indices.ndim)]
        pad_width[direction] = (pad_size, 0)
        if type_edge0 == 'edge':
            pad1 = np.pad(pad1b, pad_width, mode='edge')
        elif type_edge0 == 'symmetric':
            pad1 = np.pad(pad1b, pad_width, mode='symmetric')
        elif isinstance(type_edge0, Number):
            value = type_edge0
            pad1 = np.pad(pad1b, pad_width, mode='constant', constant_values=0)

            n_range = [np.arange(max(1, s)) for s in padded_sizes]
            n_range[direction] = np.arange(pad_size)

            el_nos = np.meshgrid(*n_range, indexing='ij')
            self.override_padded_values(tuple(el_nos), value)
        else:
            pad1 = pad1b
        return pad1

    @property
    def padded_domain(self):
        domain_sizes = [self.domain.nelx, self.domain.nely, self.domain.nelz]
        nx, ny, nz = [n + 2*p for n, p in zip(domain_sizes, self.pad_sizes)]
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
        el_x, el_y, el_z = np.meshgrid(xrange, yrange, zrange, indexing='ij')
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
        delemx = min(nx, int((radius-1e-10*dx)/dx))
        delemy = min(ny, int((radius-1e-10*dy)/dy))
        delemz = min(nz, int((radius-1e-10*dz)/dz))
        xrange = np.arange(-delemx, delemx+1)*dx
        yrange = np.arange(-delemy, delemy+1)*dy
        zrange = np.arange(-delemz, delemz+1)*dz
        coords_x, coords_y, coords_z = np.meshgrid(xrange, yrange, zrange, indexing='ij')
        self.weights = np.maximum(0.0, radius - np.sqrt(coords_x*coords_x + coords_y*coords_y + coords_z*coords_z))
        self.weights /= np.sum(self.weights)  # Volume preserving

    def _response(self, x):
        xpad = self.get_padded_vector(x)
        y3d = convolve(xpad, self.weights, mode='valid')
        y = np.zeros_like(x)
        np.add.at(y, self.el3d_orig, y3d)
        return y

    def _sensitivity(self, dfdv):
        dx3d = correlate(dfdv[self.el3d_orig], self.weights, mode='full')
        for index, _ in self.overrides:
            dx3d[index] = 0
        dx = np.zeros_like(self.sig_in[0].state)
        np.add.at(dx, self.el3d_pad, dx3d)
        return dx


class Filter(Module):
    r""" Abstract base class for any linear filter with normalization

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

    Keyword Args:
        nonpadding (numpy.array[int]): An array with indices at places where
          :math:`s_i = \max(\mathbf{s}) \: \forall\: i \notin \mathcal{N}`. For a density filter this mimics having values
          of `0` outside of the domain, thus emulating padding of the boundaries.
    """
    def _prepare(self, *args, nonpadding=None, **kwargs):
        self.H = self._calculate_h(*args, **kwargs).tocsc()

        self.Hs = self.H.sum(1)

        if nonpadding is not None:
            inds = ~np.isin(np.arange(len(self.Hs)), nonpadding)
            self.Hs[inds] = np.max(self.Hs)

    @staticmethod
    def _calculate_h(*args, **kwargs):
        r""" This method should be overridden by any child-classes to implement their own filtering behavior

        Returns:
            Filtering matrix :math:`\mathbf{H}`, *e.g.* in COO-format
        """
        raise NotImplementedError("Filter not implemented.")

    def _response(self, x):
        return np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

    def _sensitivity(self, dfdy):
        return np.asarray(self.H * (dfdy[np.newaxis].T / self.Hs))[:, 0]


class DensityFilter(Filter):
    r""" Standard density filter for a structured mesh in topology optimization

    The filtered densities are calculated as

    :math:`y_i = \sum_j \frac{H_{ij}}{s_i}x_j`,

    where :math:`H_{ij}=\max \left( r - \sqrt{ (x_j - x_i)^2 + (y_j - y_i)^2 + (z_j - z_i)^2 } , 0 \right)`,

    and :math:`s_i=\sum_j H_{ij}`.

    Input Signal:
        - ``x``: The unfiltered field :math:`\mathbf{x}`

    Output Signal:
        - ``y``: Filtered field :math:`\mathbf{y}`

    Args:
        domain: The domain layout

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
        """ Density filter: Build (and assemble) the index+data vectors for the coo matrix format
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
        xinds, yinds, zinds = np.meshgrid(xrange, yrange, zrange, indexing='ij')

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
            elcomp = els[xlow[el]:xupp[el]+1, ylow[el]:yupp[el]+1, zlow[el]:zupp[el]+1].flatten()

            indstart = ncum[el-1] if el > 0 else 0
            h_rows[indstart:ncum[el]] = el
            h_cols[indstart:ncum[el]] = elcomp

        # Calculate element distances
        dx = ix[h_rows] - ix[h_cols]
        dy = iy[h_rows] - iy[h_cols]
        dz = iz[h_rows] - iz[h_cols]

        h_values = np.maximum(0.0, radius - np.sqrt(dx*dx + dy*dy + dz*dz))

        # Finalize assembly
        return coo_matrix((h_values, (h_rows, h_cols)), shape=(nel, nel))


class OverhangFilter(Module):
    r""" Implementation of overhang filter by Langelaar (2016, 2017)

    It proceeds layer by layer through the entire domain. For each element in the current layer, the maximum printable
    density is determined by a smooth maximum of the supporting elements
    :math:`s_i = \text{smax}(\mathbf{y}_\text{supp})`. Then, the final printed density value is obtained by a smooth
    minimum operation of the desired density
    :math:`x_i` and the maximum printable density :math:`x_i` as :math:`y_i = \text{smin}(x_i, s_i)`.

    Input Signal:
        - ``x``: The unfiltered field :math:`\mathbf{x}`

    Output Signal:
        - ``y``: Filtered field :math:`\mathbf{y}`, without overhangs

    Args:
        domain: The domain layout

    Keyword Args:
        direction: Print direction as array or string, e.g. ``[0, -1]`` (in 2D) or ``"y-"`` for negative y direction.
          Currently, only directions aligned with one of the Cartesian axes are supported. Default is ``[0, 1, 0]``
        xi_0: Density value for which zero overshoot is required ( ``0 <= xi_0 <= 1`` ). Default is ``0.5``
        p: Exponent of the smooth maximum function ( ``p > 0`` ). Higher p increases accuracy, but reduces smoothness.
          Default is ``40.0``
        eps: Smooth minimum regularization parameter ( ``eps >= 0`` ). Lower eps increases accuracy, but reduces
          smoothness. Default is ``1e-4``
        nsampling: ``3`` for 2D overhang, ``5`` or ``9`` for 3D overhang. Default is ``3`` in 2D and ``5`` in 3D

    References:
      - Langelaar, M. (2017). *An additive manufacturing filter for topology optimization of print-ready designs*.
        Structural and Multidisciplinary Optimization, 55(3), 871–883.
        `doi: 10.1007/s00158-016-1522-2 <https://doi.org/10.1007/s00158-016-1522-2>`_
      - Langelaar, M. (2016). *Topology optimization of 3D self-supporting structures for additive manufacturing*.
        Additive Manufacturing, 12, 60–70.
        `doi: 10.1016/j.addma.2016.06.010 <https://doi.org/10.1016/j.addma.2016.06.010>`_
    """
    def _prepare(self,
                 domain: DomainDefinition,
                 direction=(0.0, 1.0, 0.0),
                 xi_0: float = 0.5,
                 p: float = 40.0,
                 eps: float = 1e-4,
                 nsampling: int = None):

        # Parse print direction
        if isinstance(direction, str):
            # Print axis
            axes = np.argwhere([a in direction.lower() for a in ['x', 'y', 'z']]).flatten()
            if axes.size != 1:
                raise ValueError(f"Wronly specified print direction {direction}, should be e.g. \"+x\", \"-y\"")

            # Print direction
            direction = [0.0, 0.0, 0.0]
            direction[axes[0]] = -1.0 if '-' in direction else +1.0
        direction = np.asarray(direction, dtype=np.float64).flatten()
        if direction.size < 3:
            direction = np.pad(direction, (0, 3-direction.size), 'constant', constant_values=0.0)
        elif direction.size > 3:
            direction = direction[:3]

        self.direction = direction / np.linalg.norm(direction)
        self.domain = domain
        if self.domain.dim == 2:
            assert self.direction[2] == 0.0, "Z-direction must be zero for 2-dimensional domain"
        assert abs(self.direction).sum() >= 1.0 - 1e-10, "The print direction must be aligned with either x, y or z directions"

        if nsampling is None:
            nsampling = 3 if self.domain.dim == 2 else 5
        assert (self.domain.dim == 2 and nsampling == 3) or (self.domain.dim == 3 and (nsampling == 5 or nsampling == 9)), \
            "Only 3 (2D), 5, or 9 (3D) support points supported"

        # Parameters
        self.xi_0 = xi_0
        self.p = p
        self.eps = eps
        self.nsampling = nsampling
        self.q, self.shift, self.backshift = None, None, None
        self.smax = None

    def set_parameters(self, typ: np.dtype):
        """ Set the internal smooth-maximum and smooth-minimum parameters according to the values in Langelaar, 2017

        Args:
            typ: The numeric type used in the calculations
        """
        dbl_min = np.finfo(typ).tiny
        self.q = self.p + np.log(1.0*self.nsampling) / np.log(self.xi_0)
        self.shift = 100.0 * pow(dbl_min, 1.0/self.p)  # Small shift to prevent division by 0
        self.backshift = pow(self.nsampling, 1/self.q)*pow(self.shift, self.p/self.q)*0.95  # 5% smaller to be on the safe side

    def _response(self, x):
        if self.q is None:  # Set parameters according to data type of x
            self.set_parameters(x.dtype)
        xprint = x.copy()
        self.smax = x.copy()

        # Size of the domain
        size = [self.domain.nelx, self.domain.nely, max(self.domain.nelz, 1)]

        dir_layer = int(np.argmax(abs(self.direction)))  # The axis of the print direction
        dx_layer = int(np.sign(self.direction[dir_layer]))  # Iteration direction
        ind_layer = 1 if dx_layer >= 0 else size[dir_layer]-2  # Starting index

        dir_orth1 = (dir_layer + 1) % 3
        dir_orth2 = (dir_layer + 2) % 3
        if dir_orth1 == 2 and self.domain.dim == 2:
            dir_orth1, dir_orth2 = dir_orth2, dir_orth1  # Make sure the z-direction is last for 2D

        #  Select layer offsets from:
        #                |<-- 3x 2D supports -->|<- 5-point 3D ->|<--         9-point 3D          -->|
        layer_offsets = [[-1, 0], [0, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]][:self.nsampling]

        entire_layer = np.meshgrid(range(size[dir_orth1]), range(size[dir_orth2]), indexing='ij')
        # Create masks since not all elements may be inside the domain
        support_idx = [None for _ in range(self.nsampling)]
        offset_masks = [None for _ in range(self.nsampling)]
        for i, offset in enumerate(layer_offsets):
            support_idx[i] = (entire_layer[0] + offset[0], entire_layer[1] + offset[1])
            offset_masks[i] = np.logical_and((support_idx[i][0] >= 0) * (support_idx[i][0] < size[dir_orth1]),
                                             (support_idx[i][1] >= 0) * (support_idx[i][1] < size[dir_orth2]))

        # Loop over all the layers
        while 0 <= ind_layer < size[dir_layer]:
            # 1) Get all support values
            keep = np.zeros_like(entire_layer[0], dtype=x.dtype)
            for i, (supp_idx, supp_mask) in enumerate(zip(support_idx, offset_masks)):
                el = [None, None, None]
                el[dir_layer] = ind_layer-dx_layer
                el[dir_orth1] = supp_idx[0][supp_mask]
                el[dir_orth2] = supp_idx[1][supp_mask]
                els = self.domain.get_elemnumber(*el)
                keep[supp_mask] += np.power(xprint[els]+self.shift, self.p)

            # 2) Take smooth maximum
            max_supp = np.power(keep, 1/self.q)-self.backshift
            # max_supp = np.maximum.reduce(supp_vals)  # Absolute maximum

            # 3) Take smooth minimum
            el = [None, None, None]
            el[dir_layer] = ind_layer
            el[dir_orth1] = entire_layer[0]
            el[dir_orth2] = entire_layer[1]
            els = self.domain.get_elemnumber(*el)
            self.smax[els] = max_supp  # Save maximum printable densities for sensitivity
            r1 = x[els] - max_supp
            xprint[els] = (x[els] + max_supp - np.sqrt(r1*r1 + self.eps) + np.sqrt(self.eps))/2
            # xprint[els] = np.minimum(x[els], max_supp)

            ind_layer += dx_layer

        return xprint

    def _sensitivity(self, dxprint):
        x = self.sig_in[0].state
        xprint = self.sig_out[0].state
        dx = np.zeros_like(dxprint)

        # Size of the domain
        size = [self.domain.nelx, self.domain.nely, max(self.domain.nelz, 1)]

        dir_layer = int(np.argmax(abs(self.direction)))  # The axis of the print direction
        dx_layer = int(np.sign(self.direction[dir_layer]))  # Iteration direction
        ind_layer = size[dir_layer]-1 if dx_layer >= 0 else 0  # Starting index (="ending" in response)

        dir_orth1 = (dir_layer + 1) % 3
        dir_orth2 = (dir_layer + 2) % 3
        if dir_orth1 == 2 and self.domain.dim == 2:
            dir_orth1, dir_orth2 = dir_orth2, dir_orth1  # Make sure the z-direction is last for 2D

        #  Select layer offsets from:
        #                |<-- 3x 2D supports -->|<- 5-point 3D ->|<--         9-point 3D          -->|
        layer_offsets = [[-1, 0], [0, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]][:self.nsampling]

        entire_layer = np.meshgrid(range(size[dir_orth1]), range(size[dir_orth2]), indexing='ij')
        support_idx = [None for _ in range(self.nsampling)]
        offset_masks = [None for _ in range(self.nsampling)]
        for i, offset in enumerate(layer_offsets):
            support_idx[i] = (entire_layer[0] + offset[0], entire_layer[1] + offset[1])
            offset_masks[i] = np.logical_and((support_idx[i][0] >= 0) * (support_idx[i][0] < size[dir_orth1]),
                                             (support_idx[i][1] >= 0) * (support_idx[i][1] < size[dir_orth2]))

        # Loop over all the layers
        while True:
            # 3) Take smooth minimum
            el = [None, None, None]
            el[dir_layer] = ind_layer
            el[dir_orth1] = entire_layer[0]
            el[dir_orth2] = entire_layer[1]
            els = self.domain.get_elemnumber(*el)

            # xprint[els] = (x[els] + max_supp - np.sqrt(r1*r1 + self.eps) + np.sqrt(self.eps))/2
            r1 = x[els] - self.smax[els]
            dfdr1 = -dxprint[els] * r1 / (2*np.sqrt(r1*r1+self.eps))
            dx[els] = dxprint[els]/2 + dfdr1
            dfdsmax = dxprint[els]/2 - dfdr1

            # 2) Take smooth maximum
            # max_supp = np.power(keep, 1/self.q)-self.backshift
            keep = np.power(self.smax[els] + self.backshift, self.q)
            dfdkeep = dfdsmax * np.power(keep, (1/self.q) - 1) / self.q
            c = self.p*dfdkeep

            # 1) Get all support values
            for i, (supp_idx, supp_mask) in enumerate(zip(support_idx, offset_masks)):
                el = [None, None, None]
                el[dir_layer] = ind_layer-dx_layer
                el[dir_orth1] = supp_idx[0][supp_mask]
                el[dir_orth2] = supp_idx[1][supp_mask]
                els = self.domain.get_elemnumber(*el)
                #  keep[supp_mask] += np.power(xprint[els]+self.shift, self.p)
                dxprint[els] += c[supp_mask]*np.power(xprint[els]+self.shift, self.p-1)

            ind_layer -= dx_layer
            if not 1 <= ind_layer < size[dir_layer]-1:
                break

        # Base layer is directly transferred
        el = [None, None, None]
        el[dir_layer] = ind_layer
        el[dir_orth1] = entire_layer[0]
        el[dir_orth2] = entire_layer[1]
        els = self.domain.get_elemnumber(*el)
        dx[els] = dxprint[els]
        return dx
