from pymoto import Module
import numpy as np


class Scaling(Module):
    r"""Scales (scalar) input for different response functions in optimization (objective / constraints).
    This is useful, for instance, for MMA where the objective must be scaled in a certain way for good convergence.

    Objective scaling using absolute value or vector norm (`minval` and `maxval` are both undefined):
    :math:`y^{(i)} = s \frac{x^{(i)}}{||x^{(0)}||}`

    For the constraints, the negative null-form convention is used, which means the constraint is :math:`y(x) \leq 0`.

    Upper limit constraint :math:`x\leq x_\text{max}` (`maxval` is specified):
    :math:`y = s \left( \frac{x - x_\text{max}}{|x_\text{max}|} \right)`

    Lower limit constraint :math:`x\geq x_\text{min}` (`minval` is specified):
    :math:`y = s \left( \frac{x_\text{min} - x}{|x_\text{min}|} \right)`

    Note: If the supplied minimum or maximum value is equal to zero, the normalization will be skipped.

    Input Signal:
        - ``x``: Unscaled variable :math:`x`

    Output Signal:
        - ``y``: Scaled variable :math:`y`
    """

    def __init__(self, scaling: float = 100.0, minval: float = None, maxval: float = None):
        """Initialize the scaling module

        Args:
            scaling (float, optional): Value :math:`s` to scale with. Defaults to 100.0.
            minval (float, optional): Minimum value :math:`x_\text{min}` for negative-null-form constraint
            maxval (float, optional): Maximum value :math:`x_\text{max}` for negative-null-form constraint
        """
        self.minval = minval
        self.maxval = maxval
        self.scaling = scaling
        if self.minval is not None and self.maxval is not None:
            raise RuntimeError("Only one-sided constraints are allowed. Either provide only minval or only maxval.")
        # In case of constraints, initial x-value is not required for scaling-factor
        if self.minval is not None or self.maxval is not None:
            self.sf = self.scaling
        else:
            self.sf = None

    def __call__(self, x):
        if self.sf is None:
            self.sf = self.scaling / np.linalg.norm(x)
        if self.minval is not None:
            nrm = 1 if self.minval == 0 else abs(self.minval)
            g = (self.minval - x) / nrm
        elif self.maxval is not None:
            nrm = 1 if self.maxval == 0 else abs(self.maxval)
            g = (x - self.maxval) / nrm
        else:
            g = x
        return g * self.sf

    def _sensitivity(self, dy):
        dg = dy * self.sf
        if self.minval is not None:
            nrm = 1 if self.minval == 0 else abs(self.minval)
            return -dg / nrm
        elif self.maxval is not None:
            nrm = 1 if self.maxval == 0 else abs(self.maxval)
            return dg / nrm
        else:
            return dg
