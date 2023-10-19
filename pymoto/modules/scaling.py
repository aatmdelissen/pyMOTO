from pymoto import Module


class Scaling(Module):
    r""" Scales (scalar) input for different response functions in optimization (objective / constraints).
    This is useful, for instance, for MMA where the objective must be scaled in a certain way for good convergence.

    Objective scaling (`minval` and `maxval` are both undefined):
    :math:`y^{(i)} = s \frac{x^{(i)}}{x^{(0)}}`

    For the constraints, the negative null-form convention is used, which means the constraint is :math:`y(x) \leq 0`.

    Upper limit constraint :math:`x\leq x_\text{max}` (`maxval` is specified):
    :math:`y = s \left( \frac{x}{x_\text{max}} - 1 \right)`

    Lower limit constraint :math:`x\geq x_\text{min}` (`minval` is specified):
    :math:`y = s \left( 1 - \frac{x}{x_\text{min}} \right)`

    Input Signal:
        - ``x``: Unscaled variable :math:`x`

    Output Signal:
        - ``y``: Scaled variable :math:`y`

    Keyword Args:
        scaling: Value :math:`s` to scale with
        minval: Minimum value :math:`x_\text{min}` for negative-null-form constraint
        minval: Maximum value :math:`x_\text{max}` for negative-null-form constraint
    """
    def _prepare(self, scaling: float = 100.0, minval: float = None, maxval: float = None):
        self.minval = minval
        self.maxval = maxval
        self.scaling = scaling
        if self.minval is not None and self.maxval is not None:
            raise RuntimeError("Only one-sided constraints are allowed. Either provide only minval or only maxval.")
        # In case of constraints, initial x-value is not required for scaling-factor
        if self.minval is not None or self.maxval is not None:
            self.sf = self.scaling

    def _response(self, x):
        if not hasattr(self, 'sf'):
            self.sf = self.scaling/x
        if self.minval is not None:
            g = 1 - x/self.minval
        elif self.maxval is not None:
            g = x/self.maxval - 1
        else:
            g = x
        return g * self.sf

    def _sensitivity(self, dy):
        dg = dy * self.sf
        if self.minval is not None:
            return - dg / self.minval
        elif self.maxval is not None:
            return dg / self.maxval
        else:
            return dg

