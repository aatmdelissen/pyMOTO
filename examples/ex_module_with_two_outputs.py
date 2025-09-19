"""Custom: Module with two outputs
==================================

The creation of a module with two outputs is demonstrated

Some modules may need to return two outputs (for instance, eigenvalues and eigenvectors, or absolute value and angle).
Implementation of a module that returns two values requires some special handling for the sensitivity function.
"""
import pymoto as pym


class TwoOutputs(pym.Module):
    """ This module has two inputs and two outputs """
    def __call__(self, x1, x2):
        print(f'[{type(self).__name__}]Do my response calculation')
        # Store data, which might be needed for the sensitivity calculation
        self.x1 = x1
        self.x2 = x2

        # Calculate two response values
        y1 = x1 * x2
        y2 = x1 + x2

        # Return the results
        return y1, y2

    def _sensitivity(self, df_dy1, df_dy2):
        """ This function calculate the (backward) sensitivity.
        It should handle None (zero sensitivity) as incoming adjoint variable. If both are None, the sensitivity
        will not be called.
        """
        print(f'[{type(self).__name__}]Do my sensitivity calculation')

        # Calculate the gradients with chain-rule
        # First initialize sensitivities with the correct size containing all zeros
        df_dx1 = self.x1 * 0  # The sensitivity df/dx1 is the same size as x1 (in case of a vector/matrix)
        df_dx2 = self.x2 * 0

        # In case the data of x1 and x2 were not stored, it could still be obtained here by directly accessing the state
        # of the input signals.
        also_x1 = self.sig_in[0].state
        assert also_x1 == self.x1
        also_x2 = self.sig_in[1].state
        assert also_x2 == self.x2

        # If the sensitivity of the output signal is empty, it is None. So we only need to do calculations whenever it
        # is not None. In case both sensitivities of the output signals are None, this function won't be called.
        if df_dy1 is not None:
            df_dx1 += df_dy1*self.x2
            df_dx2 += df_dy1*self.x1

        if df_dy2 is not None:
            df_dx1 += df_dy2
            df_dx2 += df_dy2

        # Return the results
        return df_dx1, df_dx2


if __name__ == "__main__":
    print(__doc__)
    print("_" * 80)
    print("-- Module setup")

    # Create signals for the inputs. The argument is the 'tag' of the signal, which is optional.
    # The tag of the signal can be seen as its name, which can be useful for printing and debugging
    x1 = pym.Signal("x1", 2.0)

    # Also create a second input signal (as our module has two inputs)
    x2 = pym.Signal("x2", 3.0)

    print(f"\nState initialized to {x1.tag} = {x1.state}, {x2.tag} = {x2.state}")

    # The module is instantiated using the constructor. In this case there is not initialization defined, so no 
    # arguments are passed.
    print("Create Module:")
    my_module = TwoOutputs()  # Module with two outputs

    print("\n-- Connect module and run forward analysis:")
    y1, y2 = my_module(x1, x2)
    y1.tag, y2.tag = 'y1', 'y2'  # Set a name for the output signals

    # The state of the output signal can be accessed using `state` again
    print(f"The result: {y1.tag} = {y1.state}, {y2.tag} = {y2.state}")

    print("\n-- Sensitivity analysis by back-propagation")
    # Calculate sensitivities
    print("\nSeed dy1/dy1 = 1.0, so we can calculate dy1/dx1 and dy1/dx2")
    # An initial 'seed' sensitivity of the response you're interested in needs to be set. We can do this by setting
    # the `sensitivity` property
    y1.sensitivity = 1.0
    my_module.sensitivity()
    # The sensitivities of the input signals can now be accessed by <Signal>.sensitivity
    print(f"dy1/d{x1.tag} = {x1.sensitivity}")
    print(f"dy1/d{x2.tag} = {x2.sensitivity}")

    # If we also want to calculate the sensitivities for the second output, we first need to reset the sensitivities
    print("\nReset sensitivities")
    my_module.reset()
    assert y1.sensitivity is None  # The sensitivity of y1 is now cleared; also those of x1 and x2

    print("\nSeed dy2/dy2 = 1.0, so we can calculate dy2/dx1 and dy2/dx2")
    y2.sensitivity = 1.0
    my_module.sensitivity()
    print(f"dy2/d{x1.tag} = {x1.sensitivity}")
    print(f"dy2/d{x2.tag} = {x2.sensitivity}")

    # You can always check your module with finite differencing
    pym.finite_difference([x1, x2], [y1, y2], random=False)
