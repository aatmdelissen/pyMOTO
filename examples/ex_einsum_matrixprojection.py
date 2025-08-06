""" 
Einstein summation: Matrix projection
=====================================

This example demonstrates how to use the `pymoto.EinSum` module to perform a matrix projection operation, specifically 
calculating V^T B V, where B is a square matrix and V is a block-vector.
"""
import pymoto as pym
import numpy as np

if __name__ == "__main__":
    print(__doc__)

    # --- INITIALIZATION ---
    
    B = pym.Signal("B")
    V = pym.Signal("V")

    N = 5  # Size
    M = 2
    B.state = np.random.rand(N, N)
    V.state = np.random.rand(N, M)
    print(f"{B.tag} = {B.state}")
    print(f"{V.tag} = {V.state}")

    # --- FORWARD ANALYSIS ---
    with pym.Network() as fn:
        # Matrix projection V^T B V
        y2 = pym.EinSum("ji, jk, kl -> il")(V, B, V)
        y2.tag = "VtBV"

    print(f"\nThe response is {y2.tag} =\n {y2.state}\n")

    # --- BACKPROPAGATION ---
    dgdy2 = np.ones_like(y2.state)  # Seed the output sensitivity
    y2.sensitivity = dgdy2
    fn.sensitivity()  # Run the backpropagation
    print("\nThe sensitivities are:")
    print(f"dg/d{B.tag} = \n{B.sensitivity}")
    print(f"dg/d{V.tag} = \n{V.sensitivity}")

    # --- Finite difference check ---
    pym.finite_difference([B, V], y2)
