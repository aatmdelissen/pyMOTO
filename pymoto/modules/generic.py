""" Generic modules, valid for general mathematical operations """
import numpy as np
from pymoto.core_objects import Module
from pymoto.utils import _concatenate_to_array, _split_from_array
try:
    from opt_einsum import contract as einsum  # Faster einsum
except ModuleNotFoundError:
    from numpy import einsum


class MathGeneral(Module):
    """ General mathematical expression module

    This block can evaluate symbolic mathematical expressions. The derivatives are automatically calculated using
    ``sympy``. Variables in this expression can be given by using the signal's global name (:attr:`Signal.tag`), or by
    the signal's position in the input signal list: ``"inp0"`` for the first input signal given, ``"inp1"`` for the
    second input signal, ``"inp2"`` for the third, *etc*...

    Example:
        Two scalars::

            from pymoto import Signal, MathGeneral
            s1 = Signal("x", 1.3)
            s2 = Signal("y", 4.8)
            m = MathGeneral([s1, s2], Signal("output"), "inp0*inp1")  # or "x * y" as expression
            m.response()
            assert (m.sig_out[0].state == 1.3*4.8)

        Scalar and Vector::

            s1 = Signal("x", np.array([1.3, 2.5, 9.4]))
            s2 = Signal("y", 4.8)
            m = MathGeneral([s1, s2], Signal("output"), "sin(x)*y").response()
            assert (m.sig_out[0].state.shape == (3,))
            from math import sin
            assert (abs(m.sig_out[0].state[0] - sin(1.3)*4.8) < 1e-10)
            assert (abs(m.sig_out[0].state[1] - sin(2.5)*4.8) < 1e-10)
            assert (abs(m.sig_out[0].state[2] - sin(9.4)*4.8) < 1e-10)

    Input signals:
        ``*args`` (`float` or `np.ndarray`): Any number of numerical inputs which match the provided expression

    Output Signal:
        ``y`` (`float` or `np.ndarray`): Result of the mathematical operation

    Args:
        expression (str): The mathematical expression to be evaluated

    References:
      - `Sympy documentation <https://docs.sympy.org/latest/index.html>`_
    """
    def _prepare(self, expression):
        from sympy import lambdify
        from sympy.parsing.sympy_parser import parse_expr

        expression = expression.replace("^", "**").lower()  # Case insensitive

        # Replace powers
        expr = parse_expr(expression)

        # Variables
        var_names = []
        for i in range(len(self.sig_in)):
            var_names += ["inp{}".format(i), ]

        # Named variables <RHO, X, ...> are converted to <sig0, sig1, ...>
        trn = {}
        for i, s in enumerate(self.sig_in):
            if len(s.tag):
                if s.tag.lower() in var_names and s.tag.lower() in expression:
                    raise RuntimeError("Name '{}' multiple defined".format(s.tag.lower()))
                trn[s.tag.lower()] = var_names[i]

        expr = [expr.subs(trn)]
        self.f = lambdify(var_names, expr, "numpy")

        # Determine derivatives
        dx = []
        for i in range(len(self.sig_in)):
            dx += [expr[0].diff(var_names[i])]

        self.df = lambdify(var_names, dx, "numpy")

    def _response(self, *args):
        self.x = args
        return self.f(*args)

    def _sensitivity(self, df_dy):
        dg_df = self.df(*self.x)  # This could be moved to _response(): less computations but more memory usage

        # Initialize sensitivities with zeroed out memory. This should ensure identical type of state and sensitivity
        dg_dx = []
        for s in self.sig_in:
            try:
                dg_dxi = s.state.copy()
                dg_dxi[...] = 0
            except (AttributeError, TypeError):  # Not numpy or zero-dimension array
                dg_dxi = s.state * 0
            # assert (isinstance(dg_dxi, type(s.state)))
            dg_dx.append(dg_dxi)

        # Sum if input is scalar
        for i, sig in enumerate(self.sig_in):
            dg_dx_add = df_dy*dg_df[i]
            if np.isrealobj(dg_dx[i]) and np.iscomplexobj(dg_dx_add):
                dg_dx_add = np.real(dg_dx_add)

            # Add the contribution
            if (not hasattr(dg_dx[i], '__len__')) or (hasattr(dg_dx[i], 'ndim') and dg_dx[i].ndim == 0):
                # Scalar type or 0-dimensional array
                dg_dx[i] += np.sum(dg_dx_add)
            else:
                dg_dx[i] += dg_dx_add

            # assert (isinstance(dg_dx[i], type(sig.state)))
        return dg_dx


class EinSum(Module):
    """ General linear algebra module which uses the Numpy function ``einsum``

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

    Output Signal:
        ``y`` (`np.ndarray`): Result of the ``einsum`` operation

    Args:
        expression (str): The ``einsum`` expression

    References:
      - `Numpy documentation on EinSum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_
      - `Understanding Numpy's EinSum <https://stackoverflow.com/questions/26089893/understanding-numpys-einsum>`_
      - `EinStein summation in Numpy <https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/>`_
      - `Optimized Einsum opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_
    """
    def _prepare(self, expression: str):
        self.expr = expression
        cmd = self.expr.split("->")
        self.indices_in = [s.strip() for s in cmd[0].split(",")]
        self.indices_out = cmd[1] if "->" in self.expr else ''

    def _response(self, *args):
        return [einsum(self.expr, *args, optimize=True)]

    def _sensitivity(self, df_in):
        n_in = len(self.sig_in)

        if (self.indices_out == '') and n_in == 1:
            # In case expression has only one input and scalar output, e.g. "i->", "ij->", the output size cannot
            # be deducted. Therefore, we add these exceptions
            if len(set(self.indices_in[0])) < len(self.indices_in[0]):
                # exception for repeated indices (e.g. trace, diagonal summing)
                if self.sig_in[0].state.ndim > 2:
                    raise TypeError(
                        "Sensitivities for repeated incides '{}' not supported for any other than trace 'ii->'."
                        .format(self.expr))
                mat = np.zeros_like(self.sig_in[0].state)
                np.fill_diagonal(mat, 1.0)
            else:
                mat = np.ones_like(self.sig_in[0].state)
            return df_in * mat

        for ind_in in self.indices_in:
            if len(set(ind_in)) < len(ind_in):
                raise TypeError("Sensitivities for repeated incides '{}' not supported for any other than trace 'ii->'."
                                .format(self.expr))

        df_out = []
        for ar in range(n_in):
            ind_in = [self.indices_out]
            ind_in += [elem for i, elem in enumerate(self.indices_in) if i != ar]
            arg_in = [s.state for i, s in enumerate(self.sig_in) if i != ar]
            arg_complex = [np.iscomplexobj(s.state) for i, s in enumerate(self.sig_in) if i != ar]
            ind_out = self.indices_in[ar]

            op = ",".join(ind_in)+"->"+ind_out
            if not np.iscomplexobj(self.sig_in[ar].state) and np.any(arg_complex) and np.iscomplexobj(df_in):
                da_i = np.zeros_like(self.sig_in[ar].state)+0j
                einsum(op, df_in, *arg_in, out=da_i, optimize=True)
                da_i = da_i.real
            else:
                da_i = np.zeros_like(self.sig_in[ar].state)
                einsum(op, df_in, *arg_in, out=da_i, optimize=True)
            df_out.append(da_i)
        return df_out


class ConcatSignal(Module):
    """ Concatenates data of multiple signals into one big vector """
    def _response(self, *args):
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
