""" Example Einsum: Matrix projection """
from pyModular import Signal, finite_difference, EinSum
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    # Matrix projection V^T B V
    print("Matrix projection")
    B = Signal("B")
    V = Signal("V")
    y2 = Signal("VtBV")
    N = 5  # Size
    M = 2
    B.set_state(np.random.rand(N, N))
    V.set_state(np.random.rand(N, M))

    m_matproj = EinSum([V, B, V], y2, "ji, jk, kl -> il")

    m_matproj.response()

    print("The response is {0} =\n {1}".format(y2.tag, y2.get_state()))

    dgdy2 = np.ones_like(y2.get_state())
    y2.set_sens(dgdy2)
    m_matproj.sensitivity()
    print("The sensitivities are:")
    print("dg/d{0} = \n{1}".format(B.tag, B.get_sens()))
    print("dg/d{0} = \n{1}".format(V.tag, V.get_sens()))

    finite_difference(m_matproj)
