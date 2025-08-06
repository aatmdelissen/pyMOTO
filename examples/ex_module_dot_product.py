"""
Vector dot product 
==================

In this example, a custom module for the vector-vector dot-product is implemented. The same behavior can be realized 
with the `pymoto.EinSum` Module, which relies on the numpy function einsum.
"""
import pymoto as pym
import numpy as np


class MyDotProduct(pym.Module):
    """ Calculates vector dot-product
    y = u . v
    """
    def __call__(self, vec1, vec2):
        return np.dot(vec1, vec2)

    def _sensitivity(self, dfdy):
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
