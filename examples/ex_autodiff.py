"""AutoMod: Automatic differentiation
=====================================

Automatically generate sensitivities using automatic differentiation

Using automatic differentiation (:py:mod:`jax` or :py:mod:`autograd`) the derivatives are generated of a user-defined 
function. Sensitivities do not need to be implemented by hand, but are automatically generated using the 
:py:class:`pymoto.AutoMod` module.
"""
import numpy as np
import pymoto as pym


def my_new_function(x1, A, x2, val=1.3):
        """ User-defined functionality which is to be differentiated.
        This function calculates a response based on a multiple input values, here for example 3.
        It is possible to use both vector and scalar inputs, as well as complex numbers.

        Args:
            x1: First vector
            A: Matrix
            x2: Second vector

        Returns:
            The results of the calculation
        """
        # E.g. test for incorrect data
        if x1 is None or x2 is None:
            raise RuntimeError("You forgot to set x1 and/or x2")

        # Calculate response
        v = x1 @ (A @ x2) + x1 * x2 + val

        # Return the results
        return v


if __name__ == "__main__":
    print(__doc__)
    print("_" * 80)

    # Initialize signals we want to use in our program
    x1 = pym.Signal("x1")
    A = pym.Signal("A")
    x2 = pym.Signal("x2")

    which = 'vector'  # Choose 'scalar' or 'vector' to test the module
    complex = True  # Choose True or False to test complex numbers
    if which == 'scalar':
        x1.state = 2.0
        x2.state = 3.0
        if complex:
            x1.state = x1.state + 1.0*1j
            x2.state = x2.state + 4.0*1j
    elif which == 'vector':
        x1.state = np.random.rand(4)
        x2.state = np.random.rand(4)
        A.state = np.random.rand(4, 4)
        if complex:
            x1.state = x1.state + 1j*np.random.rand(4)
            x2.state = x2.state + 1j*np.random.rand(4)
            A.state = A.state + 1j*np.random.rand(4, 4)

    # The module can be instantiated using the constructor
    y = pym.AutoMod(my_new_function)(x1, A, x2)
    y.tag = "y"
    print(f"The response is {y.tag} = {y.state}")

    # Check the response values; they are the same as the original function
    y_chk = my_new_function(x1.state, A.state, x2.state)
    print(f"The expected response is = {y_chk}")
    assert np.allclose(y.state, y_chk)
     
    # Check the sensitivities; these are automatically calculated using the autodiff module
    pym.finite_difference([x1, A, x2], y)
