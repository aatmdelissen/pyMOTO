from pymodular import Module, Signal, finite_difference
import jax
from jax.config import config
config.update("jax_enable_x64", True)  # Use double64 instead of the standard float32
import numpy as np
import scipy.sparse as sp


class MyNewModule(Module):
    """ This is my module that does awesome stuff.

    The (Module) is required for your module to be a module, with correct behaviour.
    This example is a module with two inputs and two outputs.
    """
    def _prepare(self, val):
        """
        This function is called by the initialization of the superclass. It can use the input parameters entered into
        the Module.create() method.
        """
        self.vjp_fn = None
        self.val = val

    def _response(self, x1, A, x2):
        """
        This function calculates a response based on a multiple input values, here for example 2. Multiple outputs can
        easily be added. Also, different response behaviours can be implemented, based on the number of inputs (function
        overloading). The 'self' object can be used to save state variables.

        :param x1: First vector
        :param A: Matrix
        :param x2: Second vector
        :return: The results of the calculation
        """
        # Incorrect data
        if x1 is None or x2 is None:
            raise RuntimeError("You forgot to set {} and {}".format(self.sig_in[0].tag, self.sig_in[1].tag))

        # Calculate response
        v1 = x1 * x2
        v2 = x1 @ (A @ x2.conj()) + v1 + self.val

        # Return the results
        return v1, v2

    def response(self):
        # Calculate the response and tangent operator (JAX Vector-Jacobian product)
        y, self.vjp_fn = jax.vjp(self._response, *[s.state for s in self.sig_in])

        # Assign all the states
        for i, s in enumerate(self.sig_out):
            s.state = y[i]

    def sensitivity(self):
        # Gather the output sensitivities
        dfdv = [s.sensitivity for s in self.sig_out]
        for i in range(len(dfdv)):
            if dfdv[i] is None:  # JAX does not accept None as 0
                dfdv[i] = np.zeros_like(self.sig_out[i].state)
            elif np.iscomplexobj(dfdv[i]):
                dfdv[i] = np.conj(dfdv[i])

        # Calculate backward sensitivity
        dfdx = self.vjp_fn(tuple(dfdv))

        # Assign the sensitivities
        for i, s in enumerate(self.sig_in):
            if np.iscomplexobj(dfdx[i]):
                s.add_sensitivity(np.conj(dfdx[i]))
            else:
                s.add_sensitivity(dfdx[i])


if __name__ == "__main__":
    print(__doc__)
    print("_" * 80)
    print("PART 1: Setup")

    # The print function lists all possible Module types
    print("Show all possible modules defined, it also lists the new locally defined module")
    Module.print_children()

    x1 = Signal("x1")
    A = Signal("A")
    x2 = Signal("x2")
    y1 = Signal("y1")
    y2 = Signal("y2")

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
    finite_difference(the_mod)