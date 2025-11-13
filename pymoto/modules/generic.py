"""Generic modules, valid for general mathematical operations"""
import numpy as np

from pymoto.core_objects import Module
from pymoto import DyadicMatrix
from pymoto.utils import _concatenate_to_array, _split_from_array

try:
    from opt_einsum import contract as einsum  # Faster einsum
except ModuleNotFoundError:
    from numpy import einsum


class MathExpression(Module):
    """General mathematical expression module

    This block can evaluate symbolic mathematical expressions. The derivatives are automatically calculated using
    ``sympy``. Variables in this expression can be given by using the signal's global name (:attr:`Signal.tag`), or by
    the signal's position in the input signal list: ``"inp0"`` for the first input signal given, ``"inp1"`` for the
    second input signal, ``"inp2"`` for the third, *etc*...

    Example:
        Two scalars::

            from pymoto import Signal, MathExpression
            s1 = Signal("x", 1.3)
            s2 = Signal("y", 4.8)
            m = MathExpression([s1, s2], Signal("output"), "inp0*inp1")  # or "x * y" as expression
            m.response()
            assert (m.sig_out[0].state == 1.3*4.8)

        Scalar and Vector::

            s1 = Signal("x", np.array([1.3, 2.5, 9.4]))
            s2 = Signal("y", 4.8)
            m = MathExpression([s1, s2], Signal("output"), "sin(x)*y").response()
            assert (m.sig_out[0].state.shape == (3,))
            from math import sin
            assert (abs(m.sig_out[0].state[0] - sin(1.3)*4.8) < 1e-10)
            assert (abs(m.sig_out[0].state[1] - sin(2.5)*4.8) < 1e-10)
            assert (abs(m.sig_out[0].state[2] - sin(9.4)*4.8) < 1e-10)

    Input signals:
        ``*args`` (`float` or `np.ndarray`): Any number of numerical inputs which match the provided expression

    Output signal:
        ``y`` (`float` or `np.ndarray`): Result of the mathematical operation

    References:
      - `Sympy documentation <https://docs.sympy.org/latest/index.html>`_
    """

    def __init__(self, expression: str):
        """Initialize the MathExpression module

        Args:
            expression (str): The mathematical expression to be evaluated
        """
        self.expression = expression

    def parse_expression(self, n_args: int = None):
        from sympy import lambdify
        from sympy.parsing.sympy_parser import parse_expr

        self.expression = self.expression.replace("^", "**").lower()  # Case insensitive

        # Replace powers
        expr = parse_expr(self.expression)

        # Variables
        var_names = []
        for i in range(n_args):
            var_names += ["inp{}".format(i),]

        if self.sig_in is not None:
            # Named variables e.g. <RHO, X, ...> are converted to <inp0, inp1, ...>
            replace_vars = {}
            for i, s in enumerate(self.sig_in):
                if hasattr(s, 'tag') and len(s.tag):
                    if s.tag.lower() in var_names and s.tag.lower() in self.expression:
                        raise RuntimeError(f"Name '{s.tag.lower()}' multiple defined")
                    replace_vars[s.tag.lower()] = var_names[i]

            expr = expr.subs(replace_vars)
        self.f = lambdify(var_names, expr, "numpy")

        # Determine derivatives
        dx = []
        for i in range(n_args):
            dx += [expr.diff(var_names[i])]

        self.df = lambdify(var_names, dx, "numpy")

    def __call__(self, *args):
        if not hasattr(self, "f"):
            self.parse_expression(n_args=len(args))

        self.x = args
        return self.f(*args)

    def _sensitivity(self, df_dy):
        dg_df = self.df(*self.x)  # This could be moved to _response(): less computations but more memory usage

        # Initialize sensitivities with zeroed out memory. This should ensure identical type of state and sensitivity
        dg_dx = []
        for xi in self.x:
            try:
                dg_dxi = xi.copy()
                dg_dxi[...] = 0
            except (AttributeError, TypeError):  # Not numpy or zero-dimension array
                dg_dxi = xi * 0
            # assert (isinstance(dg_dxi, type(s.state)))
            dg_dx.append(dg_dxi)

        # Sum if input is scalar
        for i, sig in enumerate(self.sig_in):
            dg_dx_add = df_dy * dg_df[i]
            if np.isrealobj(dg_dx[i]) and np.iscomplexobj(dg_dx_add):
                dg_dx_add = np.real(dg_dx_add)

            # Add the contribution according to broadcasting rules of NumPy
            # https://numpy.org/doc/stable/user/basics.broadcasting.html
            if (not hasattr(dg_dx[i], "__len__")) or (hasattr(dg_dx[i], "ndim") and dg_dx[i].ndim == 0):
                # Scalar type or 0-dimensional array
                dg_dx[i] += np.sum(dg_dx_add)
            elif dg_dx[i].shape != dg_dx_add.shape:
                # Reverse broadcast https://stackoverflow.com/questions/76002989/numpy-is-there-a-reverse-broadcast
                n_leading_dims = dg_dx_add.ndim - dg_dx[i].ndim
                broadcasted_dims = tuple(range(n_leading_dims))
                for ii in range(dg_dx_add.ndim - n_leading_dims):
                    if dg_dx[i].shape[ii] == 1 and dg_dx_add.shape[ii + n_leading_dims] != 1:
                        broadcasted_dims = (*broadcasted_dims, n_leading_dims + ii)

                dg_dx_add1 = np.add.reduce(dg_dx_add, axis=broadcasted_dims, keepdims=True)  # Sum broadcasted axis
                dg_dx[i] += np.squeeze(dg_dx_add1, axis=tuple(range(n_leading_dims)))  # Squeeze out singleton axis
            else:
                dg_dx[i] += dg_dx_add

            # assert (isinstance(dg_dx[i], type(sig.state)))
        return dg_dx


class EinSum(Module):
    """General linear algebra module which uses the Numpy function ``einsum``

    Many linear algebra multiplications can be implemented using this module:

    ====================== =========================== =========================
    **Operation**          **EinSum arguments**        **Python equivalent**
    ---------------------- --------------------------- -------------------------
    Vector sum             ``"i->", u``                ``y = sum(u)``
    Elementwise multiply   ``"i,i->i", u, v``          ``w = u * v``
    Dot product            ``"i,i->", u, v``           ``y = np.dot(u,v)``
    Outer product          ``"i,j->ij", u, v``         ``A = np.outer(u,v)``
    Matrix trace           ``"ii->", A``               ``y = np.trace(A)``
    Matrix-vector product  ``"ij,j->i", A, b``         ``x = A.dot(b)``
    Quadratic form         ``"i,ij,j->", b, A, b``     ``y = b.dot(A.dot(b))``
    Matrix-matrix product  ``"ij,ij->ij", A, B``       ``C = A * B``
    Transpose matrix prod. ``"ji,ij->ij", A, B``       ``C = A.T * B``
    Matrix projection      ``"ji,jk,kl->il", V, A, V`` ``B = V.T.dot(A.dot(V))``
    ====================== =========================== =========================

    Many more advanced operations are supported (see References), with exception of expressions with repeated indices
    (*e.g.* ``iij->ij``).

    An optimized version of ``einsum`` is available by installing the package
    `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_.

    Input signals:
        ``*args`` (`np.ndarray`): Any number of inputs that are passed to ``einsum`` and match the expression

    Output signal:
        ``y`` (`np.ndarray`): Result of the ``einsum`` operation

    References:
      - `Numpy documentation on EinSum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_
      - `Understanding Numpy's EinSum <https://stackoverflow.com/questions/26089893/understanding-numpys-einsum>`_
      - `EinStein summation in Numpy <https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/>`_
      - `Optimized Einsum opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_
    """

    def __init__(self, expression: str):
        """Initialize the EinSum module

        Args:
            expression (str): The ``einsum`` expression
        """
        self.expr = expression
        cmd = self.expr.split("->")
        self.indices_in = [s.strip() for s in cmd[0].split(",")]
        self.indices_out = cmd[1] if "->" in self.expr else ""

    def __call__(self, *args):
        return einsum(self.expr, *args, optimize=True)

    def _sensitivity(self, df_in):
        inps = self.get_input_states(as_list=True)
        n_in = len(inps)

        if (self.indices_out == "") and n_in == 1:
            # In case expression has only one input and scalar output, e.g. "i->", "ij->", the output size cannot
            # be deducted. Therefore, we add these exceptions
            if len(set(self.indices_in[0])) < len(self.indices_in[0]):
                # exception for repeated indices (e.g. trace, diagonal summing)
                if inps[0].ndim > 2:
                    raise NotImplementedError(f"Sensitivities for repeated incides in '{self.expr}' not supported.")
                mat = np.zeros_like(inps[0])
                np.fill_diagonal(mat, 1.0)
            else:
                mat = np.ones_like(inps[0])
            return df_in * mat

        for ind_in in self.indices_in:
            if len(set(ind_in)) < len(ind_in):
                raise NotImplementedError(f"Sensitivities for repeated incides in '{self.expr}' not supported.")

        df_out = []
        for ar in range(n_in):
            ind_in = [self.indices_out]
            ind_in += [elem for i, elem in enumerate(self.indices_in) if i != ar]
            arg_in = [v for i, v in enumerate(inps) if i != ar]
            ind_out = self.indices_in[ar]
            op = ",".join(ind_in) + "->" + ind_out  # Adjoint operator
            da_i = einsum(op, df_in, *arg_in, optimize=True)  # Perform adjoint einsum
            df_out.append(da_i.real if not np.iscomplexobj(inps[ar]) else da_i)
        return df_out


class ConcatSignal(Module):
    """Concatenates data of multiple signals into one big vector
    
    Input signals:
        - ``*args`` (`float` or `np.ndarray`): Any number of numerical inputs to concatenate

    Output signal:
        - ``y`` (`np.ndarray`): Concatenated vector of all input signals
    """

    def __call__(self, *args):
        state, self.cumlens = _concatenate_to_array(list(args))
        return state

    def _sensitivity(self, dy):
        dsens = [np.zeros_like(s.state) for s in self.sig_in]
        dx = _split_from_array(dy, self.cumlens)
        for i, s in enumerate(self.sig_in):
            if not isinstance(dsens[i], type(s.state)):
                dsens[i] = type(s.state)(dx[i])
                continue
            try:
                dsens[i][...] = dx[i]
            except TypeError:
                dsens[i] = type(s.state)(dx[i])
        return dsens


class SetValue(Module):
    """Sets the values of a numpy array at specified indices to a given value
    
    Input signal:
        ``x`` (`np.ndarray`): Input numpy array to modify

    Output signal:
        ``y`` (`np.ndarray`): Modified vector with specified indices set to a given value
    """

    def __init__(self, indices, value):
        """Initialize the SetValue module

        Args:
            indices (any valid `slice` type): Indices in the input vector to set to the 
              specified value
            value (`float` or `np.ndarray`): Value(s) to set at the specified indices
        """
        self.indices = indices
        self.value = value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[self.indices] = self.value
        return y

    def _sensitivity(self, dy: np.ndarray) -> np.ndarray:
        dx = dy.copy()
        dx[self.indices] = 0
        return dx


class AddMatrix(Module):
    r"""Compute linear combination of sparse matrices

    :math:`Y = \sum_i a_i \mathbf{A}_i`

    Any number of matrices can be added, as long as the input signals are in the form 
    `[scalar, matrix, scalar, matrix, ...]`.

    Input signals:
      - `a_1`: Scalar
      - `A_1`: Sparse matrix
      - `a_2` (optional): Second scalar
      - `A_2` (optional): Second matrix
        ... pairs of further scalar and matrices

    Output signal:
        `Y`: Linear combination of matrices
    """

    def __call__(self, *args):
        assert len(args)%2 == 0, "An even number of inputs must be given in the form of (a1, A1, a2, A2, ...)"
        Y = 0
        for ai, Ai in zip(args[0::2], args[1::2]):
            Y = Y + ai * Ai
        return Y

    def _sensitivity(self, dY: DyadicMatrix):
        args = self.get_input_states()
        out = []
        for ai, Ai in zip(args[0::2], args[1::2]):
            dai = dY.contract(Ai)
            if np.isreal(ai):
                dai = dai.real
            
            dAi = ai * dY
            if np.isrealobj(Ai):
                dAi = dAi.real
            
            out.append(dai.real if np.isreal(ai) else dai)
            out.append(dAi.real if np.isrealobj(Ai) else dAi)

        return out