"""Example 3: Vector dot product (and EinSum)"""
from pyModular import Module, Signal, finite_difference, EinSum
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
    u.set_state(np.array([2.1, 3.2, 4.3, 5.4]))
    v.set_state(np.array([-1.5, 3.8, 2.3, 8.5]))

    # Forward analysis
    m_dot.response()

    print("The response is u . v = dot({0}, {1}) = {2}".format(u.get_state(), v.get_state(), y.get_state()))

    # Sensitivity analysis
    y.set_sens(np.ones_like(y.get_state()))
    m_dot.sensitivity()

    print("The sensitivities are:")
    print("d{0}/d{1} = {2}".format(y.tag, u.tag, u.get_sens()))
    print("d{0}/d{1} = {2}".format(y.tag, v.tag, v.get_sens()))

    finite_difference(m_dot)

