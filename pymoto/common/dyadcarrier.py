from typing import Union, Iterable
import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from ..utils import _parse_to_list
try:  # Import fast optimized einsum
    from opt_einsum import contract as einsum
except ModuleNotFoundError:
    from numpy import einsum


def isdyad(x):
    """ Checks if argument is a ``DyadCarrier`` """
    return isinstance(x, DyadCarrier)


def isdense(x):
    """ Checks if argument is a dense ``numpy`` array """
    return isinstance(x, np.ndarray)


def isscalarlike(x):
    """ Checks if argument is either a scalar, an array scalar, or a 0-dim array """
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)


def isnullslice(x):
    """ Checks if argument is ``:`` slice """
    return isinstance(x, slice) and x == slice(None, None, None)


class DyadCarrier(object):
    """ Efficient storage for dyadic or rank-N matrix

    Stores only the vectors instead of creating a full rank-N matrix
    :math:`\mathbf{A} = \sum_k^N \mathbf{u}_k\otimes\mathbf{v}_k`
    or in index notation :math:`A_{ij} = \sum_k^N u_{ki} v_{kj}`. This saves a lot of memory for low :math:`N`.

    Args:
        u : (optional) List of vectors
        v : (optional) List of vectors (if ``u`` is given and ``v`` not, a symmetric dyad is assumed with ``v = u``)
    """

    __array_priority__ = 11.0  # For overriding numpy's ufuncs
    ndim = 2  # Number of dimensions

    def __init__(self, u: Iterable = None, v: Iterable = None):
        self.u = []
        self.v = []
        self.ulen = -1
        self.vlen = -1
        self.dtype = np.dtype('float64')  # Standard data type
        self.add_dyad(u, v)

    @property
    def shape(self):
        """ The shape of the matrix (nrows, ncols) """
        return (self.ulen, self.vlen)

    @property
    def size(self):
        """ Size of the matrix (nrows x ncols) """
        if self.ulen < 0 or self.vlen < 0:
            return 0
        else:
            return self.ulen * self.vlen

    def add_dyad(self, u: Iterable, v: Iterable = None, fac: float = None):
        """ Adds a list of vectors to the dyad carrier

        Checks for conforming sizes of `u` and `v`. The data inside the vectors are copied.

        It is possible to add higher-dimensional matrices, which are summed along all but the last dimension.
        The effect of this is that also cross-terms are added, e.g.
        adding the matrices :math:`\mathbf{U}=[\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3]`
        and :math:`\mathbf{V}=[\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3]`
        results in :math:`(\mathbf{u}_1+\mathbf{u}_2+\mathbf{u}_3)\otimes(\mathbf{v}_1+\mathbf{v}_2+\mathbf{v}_3)`
        being added to the DyadCarrier.

        Args:
            u: List of vectors
            v: List of vectors [Optional: If not given, use `v=u`]
            fac: Optional multiplication factor

        Returns:
            self
        """
        ulist = _parse_to_list(u)
        vlist = ulist if v is None else _parse_to_list(v)

        if len(ulist) != len(vlist):
            raise TypeError("Number of vectors in u ({}) and v({}) should be equal".format(len(ulist), len(vlist)))

        n = len(ulist)

        for i, ui, vi in zip(range(n), ulist, vlist):
            # Make sure they are numpy arrays
            if not isinstance(ui, np.ndarray):
                ui = np.array(ui)
            if not isinstance(vi, np.ndarray):
                vi = np.array(vi)

            # Sum if it is a block-matrix U.V^T = sum u_i.v_j^T
            if ui.ndim > 1:
                ui = ui.sum(axis=tuple(range(ui.ndim - 1)))
            elif ui.ndim == 0:
                ui = ui[np.newaxis]

            if vi.ndim > 1:
                vi = vi.sum(axis=tuple(range(vi.ndim - 1)))
            elif vi.ndim == 0:
                vi = vi[np.newaxis]

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
        return self

    def __getitem__(self, subscript):
        assert len(subscript) == self.ndim, "Invalid number of slices, must be 2"
        usub = [ui[subscript[0]] for ui in self.u]
        vsub = [vi[subscript[1]] for vi in self.v]

        is_uni_slice = isscalarlike(usub[0]) or isscalarlike(vsub[0])
        is_np_slice = isinstance(subscript[0], np.ndarray) and isinstance(subscript[1], np.ndarray)
        if is_np_slice and subscript[0].shape != subscript[1].shape:
            raise IndexError(f"shape mismatch: indexing arrays could not be broadcast together "
                             f"with shapes {subscript[0].shape} {subscript[1].shape}")
        if is_uni_slice or is_np_slice:
            res = 0
            for (ui, vi) in zip(usub, vsub):
                res += ui*vi

            return res
        else:
            return DyadCarrier(usub, vsub)

    def __setitem__(self, subscript, value):
        assert len(subscript) == self.ndim, "Invalid number of slices, must be 2"
        if value != 0.0:
            raise ValueError("Setting entries to other than 0 makes no sense")
        if not isnullslice(subscript[0]) and not isnullslice(subscript[1]):
            raise IndexError("Only full-column or full-row slices can be set, e.g. [:, 3] or [3:8, :]")
        for ui, vi in zip(self.u, self.v):
            if not isnullslice(subscript[0]):
                ui[subscript[0]] = value
            if not isnullslice(subscript[1]):
                vi[subscript[1]] = value

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
            if other.shape != self.shape and (self.size > 0 and other.size > 0):
                raise ValueError(f"Inconsistent shapes {self.shape} and {other.shape}")
            return DyadCarrier(self.u, self.v).__iadd__(other)
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return other + self.todense()
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
            return other - self.todense()
        else:
            return NotImplemented

    def __rmul__(self, other):  # other * self
        return DyadCarrier([other*ui for ui in self.u], self.v)

    def __mul__(self, other):  # self * other
        return DyadCarrier(self.u, [vi*other for vi in self.v])

    def copy(self):
        """ Returns a deep copy of the DyadCarrier """
        return DyadCarrier(self.u, self.v)

    def conj(self):
        """ Returns (a deep copied) complex conjugate of the DyadCarrier """
        return DyadCarrier([u.conj() for u in self.u], [v.conj() for v in self.v])

    @property
    def real(self):
        """ Returns a deep copy of the real part of the DyadCarrier """
        return DyadCarrier([*[u.real for u in self.u], *[-u.imag for u in self.u]], [*[v.real for v in self.v], *[v.imag for v in self.v]])

    @property
    def imag(self):
        """ Returns a deep copy of the imaginary part of the DyadCarrier """
        return DyadCarrier([*[u.real for u in self.u], *[u.imag for u in self.u]], [*[v.imag for v in self.v], *[v.real for v in self.v]])

    # flake8: noqa: C901
    def contract(self, mat: Union[NDArray, spmatrix] = None, rows: NDArray[int] = None, cols: NDArray[int] = None):
        r""" Performs a number of contraction operations using the DyadCarrier

        Calculates the result(s) of the quadratic form:
        :math:`y = \sum_k \mathbf{u}_k^{\text{T}} \mathbf{B} \mathbf{v}_k`

        Examples:
            * Contraction using identity matrix, is the sum of dot products, which is equal to the trace of the rank-N
              matrix:

              ``y = A.contract()`` equals
              :math:`y = \text{trace}(\mathbf{A}) = \sum_k \mathbf{u}_k^{\text{T}}\mathbf{v}_k`

            * Contraction using matrix ``B`` of size ``(N, M)`` calculates the quadratic form:

              ``y = A.contract(B)`` equals
              :math:`y = \sum_k \mathbf{u}_k^{\text{T}} \mathbf{B} \mathbf{v}_k`

            * Sliced contraction with sliced row, matrix ``B`` of size ``(n, M)`` and ``rows`` of size ``n``:

              ``y = A.contract(B, rows)``
              :math:`y =\sum_k \mathbf{u}[\texttt{rows}]_k^{\text{T}} \mathbf{B} \mathbf{v}_k`

            * Sliced contraction with sliced column, matrix ``B`` of size ``(N, m)`` and ``cols`` of size ``m``:

              ``y = A.contract(B, cols=cols)`` equals
              :math:`y = \sum_k \mathbf{u}_k^{\text{T}} \mathbf{B} \mathbf{v}[\texttt{cols}]_k`

            * Sliced contraction with both sliced rows and columns, matrix ``B`` of size ``(n, m)``,
              ``rows`` of size ``n``, and ``cols`` of size ``m``:

              ``y = A.contract(B, rows, cols)`` equals
              :math:`y = \sum_k \mathbf{u}[\texttt{rows}]_k^{\text{T}} \mathbf{B} \mathbf{v}[\texttt{cols}]_k`

            * Batch contraction with multiple matrices in batch mode, matrix ``B`` of size ``(P, N, M)``:

              ``y = A.contract(B)`` equals
              :math:`y_p = \sum_k \mathbf{u}_k^{\text{T}} \mathbf{B}_p \mathbf{v}_k`

            * Batch contraction with multiple slices, matrix ``B`` of size ``(n, M)``, ``rows`` of size ``(P, n)``:

              ``y = A.contract(B, rows)`` equals
              :math:`y_p = \sum_k \mathbf{u}[\texttt{rows}_p]_k^{\text{T}} \mathbf{B} \mathbf{v}_k`

            * Batch contraction with multiple matrices and slices, matrix ``B`` of size ``(P, n, M)``, ``rows`` of
              size ``(P, n)``:

              ``y = A.contract(B, rows)`` equals
              :math:`y_p = \sum_k \mathbf{u}[\texttt{rows}_p]_k^{\text{T}} \mathbf{B}_p \mathbf{v}_k`

            * Batch contraction with multiple matrices, row slice, and column slice, which is used ,`e.g.`, for finite
              element sensitivities. Matrix ``B`` is of size ``(P, n, m)``, ``rows`` of  size ``(P, n)``, and ``cols``
              of size ``(P, m)``:

              ``y = A.contract(B, rows, cols)`` equals
              :math:`y_p = \sum_k \mathbf{u}[\texttt{rows}_p]_k^{\text{T}} \mathbf{B}_p \mathbf{v}[\texttt{cols}_p]_k`

            * Batch contractions can be extended with multiple extra batch-dimensions. For instance, matrix
              ``B`` of size ``(P, Q, R, N, M)`` results in ``y`` of size ``(P, Q, R)``:

              ``y = A.contract(B)`` equals
              :math:`y_{pqr} = \sum_k \mathbf{u}_k^{\text{T}} \mathbf{B}_{pqr} \mathbf{v}_k`

        Args:
            mat: The matrix to contract with (optional)
            rows: Indices for the rows (optional)
            cols: Indices for the columns to use (optional)

        Returns:
            Contraction result
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

        if batchsize is None:
            val = 0.0
            for ui, vi in zip(self.u, self.v):
                uarg = ui if rows is None else ui[rows]
                varg = vi if cols is None else vi[cols]
                if mat is None:
                    val += uarg @ varg
                else:
                    val += uarg @ mat @ varg
            return val

        # Continue in batch mode
        batchvar = ''.join([chr(i+65) for i in range(len(batchsize))])

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
            val += einsum(expr, *argums)

        return val

    def todense(self):
        """ Returns a full (dense) matrix from the DyadCarrier matrix """
        warning_size = 100e+6  # Bytes
        if (self.shape[0]*self.shape[1]*self.dtype.itemsize) > warning_size:
            warnings.warn(f"Expanding a dyad results into a dense matrix. "
                          f"This is not advised for large matrices {self.shape}", ResourceWarning, stacklevel=2)

        val = np.zeros((max(0, self.shape[0]), max(0, self.shape[1])), dtype=self.dtype)

        for ui, vi in zip(self.u, self.v):
            val += np.outer(ui, vi)

        return val

    def iscomplex(self):
        """ Check if the DyadCarrier is of complex type """
        return np.iscomplexobj(np.array([], dtype=self.dtype))

    def diagonal(self, k: int = 0):
        """ Returns the diagonal of the DyadCarrier matrix """
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
        """ Shorthand transpose (returns deep copy) """
        return self.transpose()

    def transpose(self):
        """ Returns a deep copy of the transposed DyadCarrier matrix"""
        return DyadCarrier(self.v, self.u)

    def dot(self, other):
        """ Inner product """
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
