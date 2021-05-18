from pymodular import Module
from .assembly import DomainDefinition
import numpy as np
from scipy.sparse import coo_matrix


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
        nx = domain.nx  # Number of elements in x direction
        ny = domain.ny  # Number of elements in y direction
        nz = max(domain.nz, 1)  # Number of elements in z direction
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
