from abc import ABC

import numpy as np
from pymoto import Module
from ..utils import _parse_to_list

try:  # AutoDiff module
    import jax
    _has_jax = True
except ImportError as e:
    _jax_error = e
    _has_jax = False


class AutoMod(Module, ABC):
    """ Module that automatically differentiates the response function """

    def __init__(self, *args, **kwargs):
        if not _has_jax:
            raise ImportError(f"Could not create this object, as dependency \"jax\" cannot be found on this system. "
                              f"Import failed with error: {_jax_error}")
        super().__init__(*args, **kwargs)

    def response(self):
        # Calculate the response and tangent operator (JAX Vector-Jacobian product)
        y, self.vjp_fn = jax.vjp(self._response, *[s.state for s in self.sig_in])
        y = _parse_to_list(y)

        # Assign all the states
        for i, s in enumerate(self.sig_out):
            s.state = y[i]

    def sensitivity(self):
        # Gather the output sensitivities
        dfdv = [s.sensitivity for s in self.sig_out]
        if np.all([df is None for df in dfdv]):
            return
        for i in range(len(dfdv)):
            if dfdv[i] is None:  # JAX does not accept None as 0
                dfdv[i] = np.zeros_like(self.sig_out[i].state)

        dfdv = tuple(dfdv) if len(dfdv) > 1 else dfdv[0]

        # Calculate backward sensitivity
        dfdx = _parse_to_list(self.vjp_fn(dfdv))

        # Assign the sensitivities
        for i, s in enumerate(self.sig_in):
            if np.iscomplexobj(dfdx[i]):
                s.add_sensitivity(dfdx[i])
            else:
                s.add_sensitivity(dfdx[i])
