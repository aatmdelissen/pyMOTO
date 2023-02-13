from jax.config import config  # JAX must be installed on the system!
import numpy as np
import pymoto as pym
config.update("jax_enable_x64", True)  # Use double64 instead of the standard float32


class MyNewModule(pym.AutoMod):
    """ This is my module that does awesome stuff AND is automatically differentiated with JAX

    Since this module is inherited from ``AutoMod``, its sensitivity behavior is automatically handled by JAX
    """

    def _prepare(self, val):
        """ This function is called during initialization. It can be used to set some parameters """
        self.val = val

    def _response(self, x1, A, x2):
        """ Forward functionality of my new module
        This function calculates a response based on a multiple input values, here for example 2. Multiple outputs can
        easily be added. Also, different response behaviours can be implemented, based on the number of inputs (function
        overloading). The 'self' object can be used to save state variables.

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
        v1 = x1 * x2
        v2 = x1 @ (A @ x2.conj()) + v1 + self.val

        # Return the results
        return v1, v2


if __name__ == "__main__":
    print(__doc__)
    print("_" * 80)
    print("PART 1: Setup")

    # Initialize signals we want to use in our program
    x1 = pym.Signal("x1")
    A = pym.Signal("A")
    x2 = pym.Signal("x2")
    y1 = pym.Signal("y1")
    y2 = pym.Signal("y2")

    # The module can be instantiated using the constructor
    print("\nInstantiate directly:")
    the_mod = MyNewModule([x1, A, x2], [y1, y2], 3.8)

    # Set the initial values
    which = 'vector'
    complex = True
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
    the_mod.response()
    pym.finite_difference(the_mod)
