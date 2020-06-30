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

    Any linear algebra multiplication can be implemented using this module
    """
    def _prepare(self, **kwargs):
        self.expr = kwargs.get("expression", "")
        cmd = self.expr.split("->")
        self.indices_in = cmd[0].split(",")
        self.indices_out = cmd[1]

    def _response(self, x):
        self.inputs = x.copy()
        return [np.einsum(self.expr, *x)]

    def _sensitivity(self, df_in):
        df_out = []
        for ar in range(len(self.sig_in)):
            ind_in = [self.indices_out]
            ind_in += [elem for i, elem in enumerate(self.indices_in) if i != ar]
            arg_in = [elem for i, elem in enumerate(self.inputs) if i != ar]
            ind_out = self.indices_in[ar]
            if (self.indices_out == '') and len(self.inputs) == 1:
                mat = np.ones_like(self.inputs[ar])
                df_in_sep = df_in * mat
                ind_in = [self.indices_in[ar]]
                op = ",".join(ind_in)+"->"+ind_out
                da_i = np.einsum(op, df_in_sep, *arg_in)
            else:
                op = ",".join(ind_in)+"->"+ind_out
                da_i = np.einsum(op, df_in, *arg_in)
            df_out.append(da_i)
        return df_out


class SumVec(Module):
    def _prepare(self, **kwargs):
        self.n = 0

    def _response(self, x):
        self.x = x
        self.n = len(x)
        return [np.sum(x)]

    def _sensitivity(self, df_dy):
        return [df_dy*np.ones_like(self.x)]
