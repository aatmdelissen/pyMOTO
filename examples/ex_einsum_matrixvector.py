""" Example Einsum: Matrix-vector product """
from pyModular import Signal, finite_difference, EinSum
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    # Matrix-vector product
    print("Matrix-vector product")
    A = Signal("A")
    b = Signal("b")
    y1 = Signal("Ab")
    N = 5  # Size
    A.set_state(np.random.rand(N, N))
    b.set_state(np.random.rand(N))

    m_matvec = EinSum([A, b], y1, "ij,j->i")

    m_matvec.response()

    print("The response is {0} = {1}".format(y1.tag, y1.get_state()))

    dgdy1 = np.ones_like(y1.get_state())
    y1.set_sens(dgdy1)
    m_matvec.sensitivity()
    print("The sensitivities are:")
    print("dg/d{0} = {1}".format(A.tag, A.get_sens()))
    print("dg/d{0} = {1}".format(b.tag, b.get_sens()))

    finite_difference(m_matvec)
