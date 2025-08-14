"""EinSum: Matrix-vector product
================================

Using EinSum to perform matrix-vector product 

:math:`\mathbf{y}=\mathbf{A}\mathbf{b}`

No sensitivities need to be defined by the user, as these are automatically deduced from the requested multiplication
operation in :py:class:`pymoto.EinSum`.
"""
import pymoto as pym
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    # --- INITIALIZATION ---
    A = pym.Signal("A")
    b = pym.Signal("b")

    N = 5  # Size
    A.state = np.random.rand(N, N)
    b.state = np.random.rand(N)
    print(f"{A.tag} = {A.state}")
    print(f"{b.tag} = {b.state}")

    # --- FORWARD ANALYSIS ---
    with pym.Network() as fn:
        # Matrix-vector product
        y1 = pym.EinSum("ij,j->i")(A, b)
        y1.tag = "y1"

    print(f"The response is {y1.tag} = {y1.state}")

    # --- BACKPROPAGATION ---
    dgdy1 = np.ones_like(y1.state)  # Seed the output sensitivity
    y1.sensitivity = dgdy1
    fn.sensitivity()  # Run the backpropagation
    print("\nThe sensitivities are:")
    print(f"dg/d{A.tag} = {A.sensitivity}")
    print(f"dg/d{b.tag} = {b.sensitivity}")

    # --- Finite difference check ---
    pym.finite_difference([A, b], y1)
