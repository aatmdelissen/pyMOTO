""" Example Einsum: Matrix projection
"""
from pyModular import Signal, finite_difference, EinSum
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    # --- SETUP ---
    # Matrix projection V^T B V
    B = Signal("B")
    V = Signal("V")
    y2 = Signal("VtBV")
    N = 5  # Size
    M = 2
    B.state = np.random.rand(N, N)
    V.state = np.random.rand(N, M)
    print("{0} = {1}".format(B.tag, B.state))
    print("{0} = {1}".format(V.tag, V.state))

    m_matproj = EinSum([V, B, V], y2, "ji, jk, kl -> il")

    # --- FORWARD ANALYSIS ---
    m_matproj.response()

    print("The response is {0} =\n {1}".format(y2.tag, y2.state))

    # --- BACKPROPAGATION ---
    dgdy2 = np.ones_like(y2.state)
    y2.sensitivity = dgdy2
    m_matproj.sensitivity()
    print("The sensitivities are:")
    print("dg/d{0} = \n{1}".format(B.tag, B.sensitivity))
    print("dg/d{0} = \n{1}".format(V.tag, V.sensitivity))

    # --- Finite difference check ---
    finite_difference(m_matproj)
