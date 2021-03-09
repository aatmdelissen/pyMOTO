""" Example Einsum: Matrix-vector product
"""
from pymodular import Signal, finite_difference, EinSum
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    # --- SETUP ---
    # Matrix-vector product
    A = Signal("A")
    b = Signal("b")
    y1 = Signal("Ab")
    N = 5  # Size
    A.state = np.random.rand(N, N)
    b.state = np.random.rand(N)
    print("{0} = {1}".format(A.tag, A.state))
    print("{0} = {1}".format(b.tag, b.state))

    m_matvec = EinSum([A, b], y1, "ij,j->i")

    # --- FORWARD ANALYSIS ---
    m_matvec.response()

    print("The response is {0} = {1}".format(y1.tag, y1.state))

    # --- BACKPROPAGATION ---
    dgdy1 = np.ones_like(y1.state)
    y1.sensitivity = dgdy1
    m_matvec.sensitivity()
    print("\nThe sensitivities are:")
    print("dg/d{0} = {1}".format(A.tag, A.sensitivity))
    print("dg/d{0} = {1}".format(b.tag, b.sensitivity))

    # --- Finite difference check ---
    finite_difference(m_matvec)
