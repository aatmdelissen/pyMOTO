from typing import Union, Iterable
import warnings
import numpy as np
from .utils import _parse_to_list


def isdyad(x):
    return isinstance(x, DyadCarrier)


def isdense(x):
    return isinstance(x, np.ndarray)


def isscalarlike(x):
    """Is x either a scalar, an array scalar, or a 0-dim array?"""
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


class DyadCarrier(object):
    """ Sparse rank-N matrix
    Stores only the vectors instead of creating a rank-N matrix
    A_ij = sum_k u_{ki} v_{kj}
    """

    __array_priority__ = 11.0  # For overriding numpy's ufuncs
    ndim = 2

    def __init__(self, u: Iterable = None, v: Iterable = None):
        """ This is a class for efficient storage of dyadic / outer products.
        Two vectors u and v can construct a matrix A = uv^T, but this takes a lot of memory to store. Therefore, this
        class only stores the two vectors. Contractions B:uv^T = u^T.B.v can be calculated (

        :param u: List of vectors
        :param v: List of vectors
        """

        self.u = []
        self.v = []
        self.ulen = -1
        self.vlen = -1
        self.shape = (self.ulen, self.vlen)
        self.dtype = np.dtype('float64')  # Standard data type
        self.add_dyad(u, v)

    def add_dyad(self, u: Iterable, v: Iterable = None, fac: Union[float, None] = None):
        """ Adds a list of vectors to the dyad carrier
        Checks for conforming sizes. It is possible to add higher-dimensional vectors, which are summed along all but
        the last dimension. The effect of this is that also cross-terms e.g. (a1+a2+a3)*(b1+b2+b3) also are added.

        :param u: List of vectors
        :param v: List of vectors [Optional: If not given, use v=u]
        :param fac: Optional multiplication factor
        """
        ulist = _parse_to_list(u)
        vlist = ulist if v is None else _parse_to_list(v)

        if len(ulist) != len(vlist):
            raise TypeError("Number of vectors in u ({}) and v({}) should be equal".format(len(ulist), len(vlist)))

        n = len(ulist)

        for i, ui, vi in zip(range(n), ulist, vlist):
            if not isinstance(ui, np.ndarray):
                raise TypeError(f"Vector in dyadcarrier should be np.ndarray. Got {type(ui)} instead.")
            if not isinstance(vi, np.ndarray):
                raise TypeError(f"Vector in dyadcarrier should be np.ndarray. Got {type(vi)} instead.")

            # Sum if it is a block-matrix U.V^T = sum u_i.v_j^T
            if ui.ndim > 1:
                ui = ui.sum(axis=tuple(range(ui.ndim - 1)))

            if vi.ndim > 1:
                vi = vi.sum(axis=tuple(range(vi.ndim - 1)))

            # Update the dyadic matrix dimensions
            if self.ulen < 0:
                self.ulen = ui.shape[-1]

            if self.vlen < 0:
                self.vlen = vi.shape[-1]

            # Check dimensions
            if ui.shape[-1] != self.ulen:
                raise TypeError(f"U vector {i} of shape {ui.shape}, not conforming to dyad size {self.ulen}.")

            if vi.shape[-1] != self.vlen:
                raise TypeError(f"V vector {i} of shape {vi.shape}, not conforming to dyad size {self.vlen}.")

            # Don't add zero vectors
            if np.linalg.norm(ui) == 0 or np.linalg.norm(vi) == 0:
                continue

            # Add the vectors
            self.u.append(ui.copy() if fac is None else fac*ui)
            self.v.append(vi.copy())

            # Update the type
            self.dtype = np.result_type(self.dtype, ui.dtype)
            self.dtype = np.result_type(self.dtype, vi.dtype)

        # Update the shape
        self.shape = (self.ulen, self.vlen)

    def __getitem__(self, subscript):
        assert len(subscript) == self.ndim, "Invalid number of slices"
        usub = [ui[subscript[0]] for ui in self.u]
        vsub = [vi[subscript[1]] for vi in self.v]
        if isscalarlike(usub[0]):
            val = np.zeros_like(vsub[0])
            for ui, vi in zip(usub, vsub):
                val += ui*vi
            return val
        elif isscalarlike(vsub[0]):
            val = np.zeros_like(usub[0])
            for ui, vi in zip(usub, vsub):
                val += ui*vi
            return val
        else:
            return DyadCarrier(usub, vsub)

    def __pos__(self):
        return DyadCarrier(self.u, self.v)

    def __neg__(self):
        return DyadCarrier([-uu for uu in self.u], self.v)

    def __iadd__(self, other):
        self.add_dyad(other.u, other.v)
        return self

    def __add__(self, other):  # self + other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError('adding a nonzero scalar from a '
                                      'dyadcarrier is not supported')
        elif isdyad(other):
            if other.shape != self.shape:
                raise ValueError("Inconsistent shapes")
            return DyadCarrier(self.u, self.v).__iadd__(other)
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return other + self.expand()
        else:
            return NotImplemented

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __isub__(self, other):
        self.add_dyad(other.u, other.v, fac=-1.0)
        return self

    def __sub__(self, other):  # self - other
        return self.__add__(-other)

    def __rsub__(self, other):  # other - self
        if isscalarlike(other):
            if other == 0:
                return -self.copy()
            raise NotImplementedError('subtracting a dyadcarrier from a '
                                      'nonzero scalar  is not supported')
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return other - self.expand()
        else:
            return NotImplemented

    def __rmul__(self, other):  # other * self
        return DyadCarrier([other*ui for ui in self.u], self.v)

    def __mul__(self, other):  # self * other
        return DyadCarrier(self.u, [vi*other for vi in self.v])

    def copy(self):
        """
        :return: Copied instance
        """
        return DyadCarrier(self.u, self.v)

    def conj(self):
        """ Complex conjugate of the DyadCarrier
        :return: Complex conjugated DyadCarrier
        """
        return DyadCarrier([u.conj() for u in self.u], [v.conj() for v in self.v])

    def contract(self, mat: np.ndarray = None, rows: np.ndarray = None, cols: np.ndarray = None):
        """ Performs contraction of the dyadcarrier
        Calculates the result(s) of the quadratic form:
        y = sum_k u_k^T A v_k

        Standard use:
        Contraction using identity matrix, is the sum of dot products, which is equal to the trace of the rank-N matrix
        a.contract() <--> np.einsum('i,i->', ui, vi) <--> np.dot(ui, vi)

        Contraction using matrix
        a.contract(Mat[n, m]) <--> np.einsum('i,ij,j->', ui, mat, vi)
                              <--> np.dot(ui, A.dot(vi))

        Include slices:
        Row slice for submatrices
        a.contract(Mat[k, m], rows[k]) <--> np.einsum('i,ij,j->', ui[rows], mat, vi)
                                       <--> np.dot(ui[rows], A[rows,:].dot(vi))

        Column slice for submatrices
        a.contract(Mat[n, l], cols=cols[l]) <--> np.einsum('i,ij,j->', ui, mat, vi[cols])
                                            <--> np.dot(ui, A[:,cols].dot(vi[cols]))

        Row and column slice
        a.contract(Mat[k, l], rows[k], cols[l]) <--> np.einsum('i,ij,j->', ui[rows], mat, vi[cols])
                                                <--> np.dot(ui[rows], A[rows,cols].dot(vi[cols]))

        Batch mode:
        Multiple matrix contractions
        a.contract(Mat[B, n, m]) <--> np.einsum('i,Bij,j->B', ui, mat, vi)

        Multiple slices
        a.contract(Mat[k, m], rows[B, k]) <--> np.einsum('Bi,ij,j->B', ui[rows], mat, vi)

        Multiple matrices and corresponding slices
        a.contract(Mat[B, k, m], rows[B, k]) <--> np.einsum('Bi,Bij,j->B', ui[rows], mat, vi)

        Multiple matrices with row and column slices (e.g. used for finite element sensitivities)
        a.contract(Mat[B, k, l], rows[B, k], cols[B, l]) <--> np.einsum('Bi,Bij,Bj->B', ui[rows], mat, vi[cols])

        Multi-dimensional batch
        a.contract(Mat[B, C, D, k, l], rows[k], cols[l]) <--> np.einsum('i,BCDij,j->BCD', ui[rows], mat, vi[cols])

        :param mat: The matrix to contract with
        :param rows: Indices for the rows
        :param cols: Indices for the columns to use
        :return: Value(s) of the contraction
        """

        rowvar = 'i'
        colvar = 'i' if mat is None else 'j'
        matvar = 'ij'

        # Batch variables
        isbatchmat = mat is not None and mat.ndim > 2
        isbatchrow = rows is not None and rows.ndim > 1
        isbatchcol = cols is not None and cols.ndim > 1

        batchsize = mat.shape[:-2] if isbatchmat else None

        if isbatchrow:
            if batchsize is None:
                batchsize = rows.shape[:-1]
            elif rows.shape[:-1] != batchsize:
                raise ValueError("Batch size of rows {} not conforming to {}".format(rows.shape[:-1], batchsize))

        if isbatchcol:
            if batchsize is None:
                batchsize = cols.shape[:-1]
            elif cols.shape[:-1] != batchsize:
                raise ValueError("Batch size of cols {} not conforming to {}".format(cols.shape[:-1], batchsize))

        batchvar = '' if batchsize is None else ''.join([chr(i+65) for i in range(len(batchsize))])

        if isbatchmat:
            matvar = batchvar + matvar

        if isbatchrow:
            rowvar = batchvar + rowvar

        if isbatchcol:
            colvar = batchvar + colvar

        exprvars = (rowvar, colvar) if mat is None else (rowvar, matvar, colvar)
        expr = ','.join(exprvars) + '->' + batchvar

        val = 0.0 if batchsize is None else np.zeros(batchsize)
        for ui, vi in zip(self.u, self.v):
            uarg = ui if rows is None else ui[rows]
            varg = vi if cols is None else vi[cols]
            argums = (uarg, varg) if mat is None else (uarg, mat, varg)
            val += np.einsum(expr, *argums)

        return val

    def expand(self):
        """ Convert to a full (dense) matrix

        :return: Rank-N dyadic matrix
        """
        warning_size = 100e+6 # Bytes
        if (self.shape[0]*self.shape[1]*self.dtype.itemsize) > warning_size:
            warnings.warn("Expanding a dyad results into a dense matrix. This is not advised for large matrices {}"
                          .format(self.shape), RuntimeWarning, stacklevel=2)

        val = np.zeros((max(0, self.shape[0]), max(0, self.shape[1])), dtype=self.dtype)

        for ui, vi in zip(self.u, self.v):
            val += np.outer(ui, vi)

        return val

    def iscomplex(self):
        """ Check if the DyadCarrier is of complex type """
        return np.iscomplexobj(np.array([], dtype=self.dtype))

    def diagonal(self, k=0):
        """ Gets the diagonal of the DyadCarrier """
        if (self.shape[0] == 0) or (self.shape[1] == 0):
            return np.zeros(0, dtype=self.dtype)

        ustart = max(0, -k)
        vstart = max(0,  k)
        n = min(self.shape[0]-ustart, self.shape[1]-vstart)

        if n < 0:
            return np.zeros(0, dtype=self.dtype)

        diag = np.zeros(n, dtype=self.dtype)

        for ui, vi in zip(self.u, self.v):
            diag += ui[ustart:ustart+n] * vi[vstart:vstart+n]

        return diag

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return DyadCarrier(self.v, self.u)

    def dot(self, other):
        return self.__dot__(other)

    def __dot__(self, other):  # self.dot(other)
        if other.ndim == 2:
            return self.__matmul__(other)

        val = np.zeros_like(self.u[0])
        for ui, vi in zip(self.u, self.v):
            val += ui * vi.dot(other)
        return val

    def __rdot__(self, other):  # other.dot(self)
        if other.ndim == 2:
            return self.__rmatmul__(other)

        val = np.zeros_like(self.v[0])
        for ui, vi in zip(self.u, self.v):
            val += vi * other.dot(ui)
        return val

    def __matmul__(self, other):  # self @ other
        if other.ndim == 1:
            return self.__dot__(other)

        return DyadCarrier(self.u, [vi@other for vi in self.v])

    def __rmatmul__(self, other):  # other @ self
        if other.ndim == 1:
            return self.__rdot__(other)
        return DyadCarrier([other@ui for ui in self.u], self.v)
