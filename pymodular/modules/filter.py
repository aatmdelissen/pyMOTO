from pymodular import Module
from .assembly import DomainDefinition
import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage import convolve


class FilterConv(Module):
    def _prepare(self, domain: DomainDefinition, weights: np.ndarray, mode: str = 'reflect'):
        self.domain = domain
        self.weights = weights
        self.mode = mode

    def set_filter_radius(self, radius: float, element_units=False):
        if element_units:
            dx, dy, dz = 1.0, 1.0, 1.0
        else:
            dx, dy, dz = self.domain.element_size
        delemx, delemy = int((radius-1e-10*dx)/dx), int((radius-1e-10*dy)/dy)

        xrange = np.arange(-delemx, delemx+1)*dx
        yrange = np.arange(-delemy, delemy+1)*dy
        coords_x, coords_y = np.meshgrid(xrange, yrange)
        self.weights = np.maximum(0.0, radius - np.sqrt(coords_x*coords_x + coords_y*coords_y))
        self.weights /= np.sum(self.weights)  # Volume preserving

    def _response(self, x):
        xbox = x.reshape(self.domain.nelx, self.domain.nely, order='F').T  # TODO 3d?
        return convolve(xbox, self.weights, mode=self.mode).flatten()

    def _sensitivity(self, dfdv):
        ybox = dfdv.reshape(self.domain.nelx, self.domain.nely, order='F').T  # TODO 3d
        return convolve(ybox, self.weights, mode=self.mode).flatten()


class Filter(Module):
    def _prepare(self, *args, nonpadding=None, **kwargs):
        self.H = self.calculate_h(*args, **kwargs).tocsc()

        self.Hs = self.H.sum(1)

        if nonpadding is not None:
            inds = ~np.isin(np.arange(len(self.Hs)), nonpadding)
            self.Hs[inds] = np.max(self.Hs)

    @staticmethod
    def calculate_h(*args, **kwargs):
        raise NotImplementedError("Filter not implemented.")

    def _response(self, x):
        return np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

    def _sensitivity(self, dfdv):
        return np.asarray(self.H * (dfdv[np.newaxis].T / self.Hs))[:, 0]


class Density(Filter):
    """ Standard density filter for topology optimization
    References:
    [1] Bruns, T. E., & Tortorelli, D. A. (2001). Topology optimization of non-linear elastic structures and compliant mechanisms.
        Computer Methods in Applied Mechanics and Engineering, 190(26–27), 3443–3459. https://doi.org/10.1016/S0045-7825(00)00278-4
    [2] TODO - What is the other reference that is commonly used?
    """
    @staticmethod
    def calculate_h(domain: DomainDefinition, radius=2.0):
        """
        Filter: Build (and assemble) the index+data vectors for the coo matrix format
        Total number of filter entries - for every element, a number of entries wrt other elements are needed (not
        including boundaries)
        Takes a square domain around current element, encompassing circle with radius floor(rmin)

        :return: H
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
    """ Implementation of overhang filter according to the work of Langelaar (2016, 2017)

    2D: Langelaar, M. (2017). An additive manufacturing filter for topology optimization of print-ready designs.
        Structural and Multidisciplinary Optimization, 55(3), 871–883. https://doi.org/10.1007/s00158-016-1522-2
    3D: Langelaar, M. (2016). Topology optimization of 3D self-supporting structures for additive manufacturing.
        Additive Manufacturing, 12, 60–70. https://doi.org/10.1016/j.addma.2016.06.010
    """
    def _prepare(self,
                 domain: DomainDefinition,
                 direction=(0.0, 1.0, 0.0),
                 xi_0: float = 0.5,
                 p: float = 40.0,
                 eps: float = 1e-4,
                 nsampling: int = None):
        """
        :param domain: The domain layout
        :param direction: Print direction as array or string, e.g. [0, -1] or "y-" for negative y direction. Default is [0, 1, 0]
        :param xi_0: Density value for which zero overshoot is required ( 0 <= xi_0 <= 1 )
        :param p: Exponent of the smooth maximum function ( p > 0 ). Higher p increases accuracy, but reduces smoothness.
        :param eps: Smooth minimum regularization parameter ( eps >= 0 ). Lower eps increases accuracy, but reduces smoothness.
        :param nsampling: 3 for 2D overhang, 5 or 9 for 3D overhang
        """

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
        """ Set the parameters according to the values in Langelaar, 2017 """
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
