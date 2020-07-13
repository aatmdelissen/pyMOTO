import numpy as np
from .core_objects import Module


# ############### STANDARD LIBRARY OF BLOCKS ##################
class MathGeneral(Module):
    """ General mathematical expression module

    This block can be initialized using a symbolic expression. The derivatives are automatically
    calculated using SymPy. Variables in this expression can be given by using the signal's global name (tag), or the
    local name such as "inp0", "inp1" and "inp2" for the first three signals.
    """
    def _prepare(self, expression):
        from sympy import lambdify
        from sympy.parsing.sympy_parser import parse_expr

        # Replace powers
        expr = parse_expr(expression.replace("^", "**").lower())

        # Variables
        var_names = []
        for i in range(len(self.sig_in)):
            var_names += ["inp{}".format(i), ]

        # Named variables <RHO, X, ...> are converted to <sig0, sig1, ...>
        trn = {}
        for i, s in enumerate(self.sig_in):
            if len(s.tag):
                if s.tag.lower() in var_names:
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
        dg_df = self.df(*self.x)
        dg_dx = [df_dy*dg for dg in dg_df]

        # Sum if input is scalar
        for i, sig in enumerate(self.sig_in):
            if dg_dx[i] is None:
                continue

            state_len = 1
            if hasattr(sig.get_state(), "__len__"):
                state_len = len(sig.get_state())

            sens_len = 1
            if hasattr(dg_dx[i], "__len__"):
                sens_len = len(dg_dx[i])

            if state_len == 1 and sens_len > 1:
                dg_dx[i] = np.sum(dg_dx[i])

            if not isinstance(dg_dx[i], type(sig.get_state())):
                dg_dx[i] = np.array(dg_dx[i])

        return dg_dx


class EinSum(Module):
    """ General linear algebra module which uses the numpy function einsum

    Many linear algebra multiplications can be implemented using this module:
    Vector sum             "i->"          y = sum(u)
    Elementwise multiply   "i,i->i"       w = u * v
    Dot product            "i,i->"        y = np.dot(u,v)
    Outer product          "i,j->ij"      A = np.outer(u,v)
    Matrix-vector product  "ij,j->i"      x = A.dot(b)
    Quadratic form         "i,ij,j->"     y = b.dot(A.dot(b))
    Matrix-matrix product  "ij,ij->ij"    C = A * B
    Transpose matrix prod. "ji,ij->ij"    C = A.T * B
    Matrix projection      "ji,jk,kl->il" B = V.T.dot(A.dot(V))

    Many more advanced operations are also supported, with exception of expressions with repeated indices (e.g. iij->ij)

    More info:
    https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
    """
    def _prepare(self, expression):
        self.expr = expression
        cmd = self.expr.split("->")
        self.indices_in = [s.strip() for s in cmd[0].split(",")]
        self.indices_out = cmd[1] if "->" in self.expr else ''

    def _response(self, *args):
        return [np.einsum(self.expr, *args)]

    def _sensitivity(self, df_in):
        n_in = len(self.sig_in)

        if (self.indices_out == '') and n_in == 1:
            # In case expression has only one input and scalar output, e.g. "i->", "ij->", the output size cannot
            # be deducted. Therefore we add these exceptions
            if len(set(self.indices_in[0])) < len(self.indices_in[0]):
                # exception for repeated indices (e.g. trace, diagonal summing)
                if self.sig_in[0].get_state().ndim > 2:
                    raise TypeError(
                        "Sensitivities for repeated incides '{}' not supported for any other than trace 'ii->'."
                        .format(self.expr))
                mat = np.zeros_like(self.sig_in[0].get_state())
                np.fill_diagonal(mat, 1.0)
            else:
                mat = np.ones_like(self.sig_in[0].get_state())
            return np.conj(np.conj(df_in) * mat)

        for ind_in in self.indices_in:
            if len(set(ind_in)) < len(ind_in):
                raise TypeError("Sensitivities for repeated incides '{}' not supported for any other than trace 'ii->'."
                                .format(self.expr))

        df_out = []
        for ar in range(n_in):
            ind_in = [self.indices_out]
            ind_in += [elem for i, elem in enumerate(self.indices_in) if i != ar]
            arg_in = [s.get_state() for i, s in enumerate(self.sig_in) if i != ar]
            ind_out = self.indices_in[ar]

            da_i = np.zeros_like(self.sig_in[ar].get_state())
            op = ",".join(ind_in)+"->"+ind_out
            np.einsum(op, np.conj(df_in), *arg_in, out=da_i)
            df_out.append(np.conj(da_i))
        return df_out


class SumVec(Module):
    def _response(self, x):
        self.x = x
        return [np.sum(x)]

    def _sensitivity(self, df_dy):
        return [df_dy*np.ones_like(self.x)]

