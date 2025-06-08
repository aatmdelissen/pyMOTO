from typing import Callable
import numpy as np
from pymoto import Module
from ..utils import _parse_to_list

try:  # AutoDiff module
    import jax
    _has_jax = True
except ImportError as e:
    _jax_error = e
    _has_jax = False


class AutoMod(Module):
    """ Module that automatically differentiates the response function """

    def __init__(self, func: Callable):
        if not _has_jax:
            raise ImportError(f"Could not create this object, as dependency \"jax\" cannot be found on this system. "
                              f"Import failed with error: {_jax_error}")
        self.func = func

    def __call__(self, *args):
        # Calculate the response and tangent operator (JAX Vector-Jacobian product)
        y, self.vjp_fn = jax.vjp(self.func, *args)
        return y

    def _sensitivity(self, *dfdv):
        # Gather the output sensitivities
        if all([df is None for df in dfdv]):
            return
        for i in range(len(dfdv)):
            if dfdv[i] is None:  # JAX does not accept None as 0
                dfdv[i] = np.zeros_like(self.sig_out[i].state)

        dfdv = tuple(dfdv) if len(dfdv) > 1 else dfdv[0]

        # Calculate backward sensitivity
        return _parse_to_list(self.vjp_fn(dfdv))
