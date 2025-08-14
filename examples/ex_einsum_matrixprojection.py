"""EinSum: Matrix projection
============================

Perform matrix projection  using the generic EinSum module

This example demonstrates how to use the :py:class:`pymoto.EinSum` module to perform a matrix projection operation, 
specifically calculating 
:math:`\mathbf{A}_p = \mathbf{V}^T \mathbf{A} \mathbf{V}`, 
where :math:`\mathbf{A}` is a square matrix and :math:`\mathbf{V}` is a block-vector. This is useful for generation of 
reduced model with Galerkin projection.
"""
import pymoto as pym
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    # --- INITIALIZATION ---
    
    A = pym.Signal("A")
    V = pym.Signal("V")

    N = 5  # Size
    M = 2
    A.state = np.random.rand(N, N)
    V.state = np.random.rand(N, M)
    print(f"{A.tag} = {A.state}")
    print(f"{V.tag} = {V.state}")

    # --- FORWARD ANALYSIS ---
    with pym.Network() as fn:
        # Matrix projection V^T A V
        Ap = pym.EinSum("ji, jk, kl -> il")(V, A, V)
        Ap.tag = "A_p"

    print(f"\nThe response is {Ap.tag} =\n {Ap.state}\n")

    # --- BACKPROPAGATION ---
    dgdAp = np.ones_like(Ap.state)  # Seed the output sensitivity
    Ap.sensitivity = dgdAp
    fn.sensitivity()  # Run the backpropagation
    print("\nThe sensitivities are:")
    print(f"dg/d{A.tag} = \n{A.sensitivity}")
    print(f"dg/d{V.tag} = \n{V.sensitivity}")

    # --- Finite difference check ---
    pym.finite_difference([A, V], Ap)
