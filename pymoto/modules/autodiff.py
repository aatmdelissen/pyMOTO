from typing import Callable
import numpy as np
from pymoto import Module
from ..utils import _parse_to_list

try:  # JAX AutoDiff module
    import jax
except ImportError as e:
    _jax_error = e
    jax = None

try:  # Autograd https://github.com/HIPS/autograd
    import autograd
except ImportError as e:
    _autograd_error = e
    autograd = None


class AutoMod(Module):
    """Module that automatically differentiates the response function"""

    def __init__(self, func: Callable, backend="autograd"):
        if "autograd" in backend.lower():
            if autograd is None:
                raise ImportError(
                    f'Could not create this object, as dependency "autograd" cannot be found on this system. '
                    f"Import failed with error: {_autograd_error}"
                )
        elif "jax" in backend.lower():
            if jax is None:
                raise ImportError(
                    f'Could not create this object, as dependency "jax" cannot be found on this system. '
                    f"Import failed with error: {_jax_error}"
                )
        else:
            raise ValueError("Only `autograd` or `jax` are supported as backends")
        self.func = func
        self.backend = backend

    def __call__(self, *args):
        # Calculate the response and tangent operator (JAX Vector-Jacobian product)
        if "autograd" in self.backend.lower():
            if not hasattr(self, "vjp_generator"):
                self.vjp_generator = autograd.make_vjp(self.func, list(range(len(args))))
            self.vjp_fn, y = self.vjp_generator(*args)
        elif "jax" in self.backend.lower():
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
