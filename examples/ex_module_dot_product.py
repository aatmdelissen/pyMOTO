"""Custom: Vector dot product 
=============================

Making a custom module for the vector-vector dot-product 

:math:`y = \mathbf{u}\cdot\mathbf{v}`

In this example, a custom module for the vector-vector dot-product is implemented. The same behavior can be realized 
with the :py:class:`pymoto.EinSum` module, which relies on the :py:mod:`numpy` function :py:func:`einsum`.
"""
import pymoto as pym
import numpy as np


class MyDotProduct(pym.Module):
    """ Calculates vector dot-product
    y = u . v
    """
    def __call__(self, vec1, vec2):
        """ This is the response (forward) behavior used to calculate the dot product """
        return np.dot(vec1, vec2)

    def _sensitivity(self, dfdy):
        """ This is the sensitivity (backward) behavior used for the derivative calculation. The argument `dfdy` is the 
        sensitivity of the response `f` with respect to this module's output `y`. It returns the sensitivity of the 
        response `f` with respect to the module's inputs `vec1` and `vec2`
        """
        vec1 = self.sig_in[0].state
        vec2 = self.sig_in[1].state
        return vec2*dfdy, vec1*dfdy


if __name__ == '__main__':
    print(__doc__)

    # --- SETUP ---
    # Initialize signals
    u = pym.Signal("u", np.array([2.1, 3.2, 4.3, 5.4]))
    v = pym.Signal("v", np.array([-1.5, 3.8, 2.3, 8.5]))

    with pym.Network() as fn:
        # Initialize module and connect signals
        use_einsum = False  # Set to True to use pymoto.EinSum instead of MyDotProduct
        if use_einsum:
            y = pym.EinSum("i,i->")(u, v)
        else:
            y = MyDotProduct()(u, v)
        y.tag = "u.v"  # Set a name for the output signal

    print(f"The response is u . v = dot({u.state}, {v.state}) = {y.state}")

    # --- BACKPROPAGATION ---
    y.sensitivity = 1.0
    fn.sensitivity()

    print("\nThe sensitivities are:")
    print(f"d{y.tag}/d{u.tag} = {u.sensitivity}")
    print(f"d{y.tag}/d{v.tag} = {v.sensitivity}")

    # --- Finite difference check ---
    pym.finite_difference([u, v], y, random=False)
