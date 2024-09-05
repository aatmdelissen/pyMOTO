import warnings
import abc
import numpy as np
import scipy.special as spsp
from pymoto import Module


class AggActiveSet:
    """ Determine active set by discarding lower or upper fraction of a set of values

    Args:
       lower_rel: Fraction of values closest to minimum to discard (based on value)
       upper_rel: Fraction of values closest to maximum to discard (based on value)
       lower_amt: Fraction of lowest values to discard (based on sorting)
       upper_amt: Fraction of highest values to discard (based on sorting)
    """
    def __init__(self, lower_rel=0.0, upper_rel=1.0, lower_amt=0.0, upper_amt=1.0):
        assert upper_rel > lower_rel, "Upper must be larger than lower to keep values in the set"
        assert upper_amt > lower_amt, "Upper must be larger than lower to keep values in the set"
        self.lower_rel, self.upper_rel = lower_rel, upper_rel
        self.lower_amt, self.upper_amt = lower_amt, upper_amt

    def __call__(self, x):
        """ Generate an active set for given array """
        xmin, xmax = np.min(x), np.max(x)
        if (xmax - xmin) == 0:  # All values are the same, so no active set can be taken
            return Ellipsis

        sel = np.ones_like(x, dtype=bool)

        # Select based on value
        xrel = (x - xmin) / (xmax - xmin)  # Normalize between 0 and 1
        if self.lower_rel > 0:
            sel = np.logical_and(sel, xrel >= self.lower_rel)
        if self.upper_rel < 1:
            sel = np.logical_and(sel, xrel <= self.upper_rel)

        # Remove lowest and highest N values
        i_sort = np.argsort(x)
        if self.lower_amt > 0:
            n_lower_amt = int(x.size * self.lower_amt)
            sel[i_sort[:n_lower_amt]] = False

        if self.upper_amt < 1:
            n_upper_amt = int(x.size * (1 - self.upper_amt))
            sel[i_sort[-n_upper_amt:]] = False

        return sel


class AggScaling:
    """ Scaling strategy to absolute minimum or maximum

    Args:
        which: Scale to `min` or `max`
        damping(optional): Damping factor between [0, 1), for a value of 0.0 the aggregation approximation is corrected
          to the exact maximum or minimum of the input set
    """
    def __init__(self, which: str, damping=0.0):
        self.damping = damping
        if which.lower() == 'min':
            self.f = np.min
        elif which.lower() == 'max':
            self.f = np.max
        else:
            raise ValueError("Argument `which` can only be 'min' or 'max'")
        self.sf = None

    def __call__(self, x, fx_approx):
        """ Determine scaling factor

        Args:
            x: Set of values
            fx_approx: Approximated minimum / maximum

        Returns:
            Scaling factor
        """
        trueval = self.f(x)
        scale = trueval / fx_approx
        if self.sf is None:
            self.sf = scale
        else:
            self.sf = self.damping * self.sf + (1 - self.damping) * scale
        return self.sf


class Aggregation(Module):
    """ Generic Aggregation module (cannot be used directly, but can only be used as superclass)

    Keyword Args:
        scaling(optional): Scaling strategy to improve approximation :py:class:`pymoto.AggScaling`
        active_set(optional): Active set strategy to improve approximation :py:class:`pymoto.AggActiveSet`
    """
    def _prepare(self, scaling: AggScaling = None, active_set: AggActiveSet = None):
        # This prepare function MUST be called in the _prepare function of sub-classes
        self.scaling = scaling
        self.active_set = active_set
        self.sf = 1.0

    @abc.abstractmethod
    def aggregation_function(self, x):
        """ Calculates f(x) """
        raise NotImplementedError()

    @abc.abstractmethod
    def aggregation_derivative(self, x):
        """" Calculates df(x) / dx """
        raise NotImplementedError()

    def _response(self, x):
        # Determine active set
        if self.active_set is not None:
            self.select = self.active_set(x)
        else:
            self.select = Ellipsis

        # Get aggregated value
        xagg = self.aggregation_function(x[self.select])

        # Scale
        if self.scaling is not None:
            self.sf = self.scaling(x[self.select], xagg)
        return self.sf * xagg

    def _sensitivity(self, dfdy):
        x = self.sig_in[0].state
        dydx = self.aggregation_derivative(x[self.select])
        dx = np.zeros_like(x)
        dx[self.select] += self.sf * dfdy * dydx
        return dx


class PNorm(Aggregation):
    r""" P-norm aggregration

    :math:`S_p(x_1, x_2, \dotsc, x_n) = \left( \sum_i (|x_i|^p) \right)^{1/p}

    Only valid for positive :math:`x_i` when approximating the minimum or maximum

    Args:
        p: Power of the p-norm. Approximate maximum for `p>0` and minimum for `p<0`
        scaling(optional): Scaling strategy to improve approximation :py:class:`pymoto.AggScaling`
        active_set(optional): Active set strategy to improve approximation :py:class:`pymoto.AggActiveSet`
    """
    def _prepare(self, p=2, scaling: AggScaling = None, active_set: AggActiveSet = None):
        self.p = p
        self.y = None
        super()._prepare(scaling, active_set)

    def aggregation_function(self, x):
        if np.min(x) < 0:
            warnings.warn("PNorm is only valid for positive x")

        # Get p-norm
        return np.sum(np.abs(x) ** self.p) ** (1/self.p)

    def aggregation_derivative(self, x):
        pval = np.sum(np.abs(x) ** self.p) ** (1 / self.p - 1)
        return pval * np.sign(x) * np.abs(x)**(self.p - 1)


class SoftMinMax(Aggregation):
    r""" Soft maximum/minimum function

    :math:`S_a(x_1, x_2, \dotsc, x_n) = \frac{\sum_i (x_i \exp(a x_i))}{\sum_i (\exp(a x_i))}`

    When using as maximum, it underestimates the maximum
    It is exact however when :math:`x_1=x_2=\dotsc=x_n`

    Args:
        alpha: Scaling factor of the soft function. Approximate maximum for `alpha>0` and minimum for `alpha<0`
        scaling(optional): Scaling strategy to improve approximation :py:class:`pymoto.AggScaling`
        active_set(optional): Active set strategy to improve approximation :py:class:`pymoto.AggActiveSet`
    """
    def _prepare(self, alpha=1.0, scaling: AggScaling = None, active_set: AggActiveSet = None):
        self.alpha = alpha
        self.y = None
        super()._prepare(scaling, active_set)

    def aggregation_function(self, x):
        self.y = np.sum(x * spsp.softmax(self.alpha * x))
        return self.y

    def aggregation_derivative(self, x):
        return spsp.softmax(self.alpha * x) * (1 + self.alpha * (x - self.y))


class KSFunction(Aggregation):
    r""" Kreisselmeier and Steinhauser function from 1979

    :math:`S_\rho(x_1, x_2, \dotsc, x_n) = \frac{1}{\rho} \ln \left( \sum_i \exp(\rho x_i) \right)`

    Args:
        rho: Scaling factor of the KS function. Approximate maximum for `rho>0` and minimum for `rho<0`
        scaling(optional): Scaling strategy to improve approximation :py:class:`pymoto.AggScaling`
        active_set(optional): Active set strategy to improve approximation :py:class:`pymoto.AggActiveSet`
    """
    def _prepare(self, rho=1.0, scaling: AggScaling = None, active_set: AggActiveSet = None):
        self.rho = rho
        self.y = None
        super()._prepare(scaling, active_set)

    def aggregation_function(self, x):
        return 1/self.rho * np.log(np.sum(np.exp(self.rho * x)))

    def aggregation_derivative(self, x):
        erx = np.exp(self.rho * x)
        return erx / np.sum(erx)
