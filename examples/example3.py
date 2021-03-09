"""Example 3: Vector dot product (and EinSum)
In this example, a module for the vector-vector dot-product is implemented. The same behavior can be realized with the
EinSum Module, which relies on the numpy function einsum.
"""
from pymodular import Module, Signal, finite_difference, EinSum
import numpy as np


class MyDotProduct(Module):
    """ Calculates vector dot-product
    y = u . v
    """
    def _response(self, vec1, vec2):
        return np.dot(vec1, vec2)

    def _sensitivity(self, dfdy):
        vec1 = self.sig_in[0].get_state()
        vec2 = self.sig_in[1].get_state()
        return vec2*dfdy, vec1*dfdy


if __name__ == '__main__':
    print(__doc__)

    # --- SETUP ---
    # Initialize signals
    u = Signal("u")
    v = Signal("v")
    y = Signal("y")

    # Initialize module and connect signals
    use_einsum = True
    if use_einsum:
        m_dot = EinSum([u, v], y, expression="i,i->")
    else:
        m_dot = MyDotProduct([u, v], y)

    # Set initial values
    u.state = np.array([2.1, 3.2, 4.3, 5.4])
    v.state = np.array([-1.5, 3.8, 2.3, 8.5])

    # --- FORWARD ANALYSIS ---
    m_dot.response()

    print("The response is u . v = dot({0}, {1}) = {2}".format(u.state, v.state, y.state))

    # --- BACKPROPAGATION ---
    y.sensitivity = np.ones_like(y.state)
    m_dot.sensitivity()

    print("\nThe sensitivities are:")
    print("d{0}/d{1} = {2}".format(y.tag, u.tag, u.sensitivity))
    print("d{0}/d{1} = {2}".format(y.tag, v.tag, v.sensitivity))

    # --- Finite difference check ---
    finite_difference(m_dot)

