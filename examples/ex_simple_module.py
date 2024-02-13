""" Example: Module templates for generic modules
It shows possibilities for a generic Module, and also the importance of the sensitivity reset.
"""
from pymoto import Module, Signal, finite_difference


class MostSimple(Module):
    """ Example of a minimal module with two inputs and one output """
    def _response(self, x1, x2):
        """ Forward-path calculation is put here """
        print(f'[{type(self).__name__}] Do my response calculation')
        print(f'\tInputs are {self.sig_in[0].tag} and {self.sig_in[1].tag}')
        return x1 * x2

    def _sensitivity(self, df_dy):
        """ Backward-path sensitivity calculation here.
        In case df_dy is None, the function will automatically not be called
        """
        print(f'[{type(self).__name__}] Do my sensitivity calculation')
        x1, x2 = [s.state for s in self.sig_in]
        df_dx1 = df_dy * x2  # Apply chain rule df/dx1 = df/dy * dy/dx1
        df_dx2 = df_dy * x1
        return df_dx1, df_dx2  # Return the sensitivities with respect to input signals


class WithPrepare(Module):
    """ This module takes a parameter during initialization
    Example usage:
    >>> import pymoto as pym
    >>> x = pym.Signal('x', state=3.5)
    >>> y = pym.Signal('y')
    >>> m = WithPrepare(x, y, 1.2, optional_value='bar')
    [WithPrepare] Prepare my module
    value = 1.2
    optional_value = bar
    >>> m.response() # doctest: +ELLIPSIS
    [WithPrepare] Do my response calculation
    Message = bar, x = 3.5, y=4.2
    ...
    >>> y.state
    4.2
    """
    def _prepare(self, value, optional_value='foo'):
        """ This prepare is called during initialization of the module, and can be used for set-up """
        print(f'[{type(self).__name__}] Prepare my module')
        print('value = {}'.format(value))
        print('optional_value = {}'.format(optional_value))
        self.value = value
        self.optional_value = optional_value

    def _response(self, x):
        print(f'[{type(self).__name__}] Do my response calculation')
        y = x * self.value
        print(f"Message = {self.optional_value}, x = {x}, y={y}")
        return y

    def _sensitivity(self, df_dy):
        print(f'[{type(self).__name__}] Do my sensitivity calculation')
        return df_dy * self.value


class TwoOutputs(Module):
    """ This module has two inputs and two outputs """
    def _response(self, x1, x2):
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
        if df_dv1 is not None:
            df_dx1 += df_dv1*self.x2
            df_dx2 += df_dv1*self.x1

        if df_dv2 is not None:
            df_dx1 += df_dv2
            df_dx2 += df_dv2

        # Return the results
        return df_dx1, df_dx2


if __name__ == "__main__":
    print(__doc__)
    print("_" * 80)
    print("-- MostSimple Module: Setup")

    # Create signals for the inputs. The argument is the 'tag' of the signal, which is optional.
    # The tag of the signal can be seen as its name, which can be useful for printing and debugging
    x1 = Signal("x1")

    # Also create a second input signal (as our module has two inputs)
    x2 = Signal("x2")

    # And create a signal for the output
    y = Signal("y")

    # The module is instantiated using the constructor. The first argument is a list of input signals, and the
    # second argument is the output signal.
    print("Instantiate MostSimple Module:")
    simple_module = MostSimple([x1, x2], y)

    print("\n-- MostSimple Module: Forward analysis")
    try:
        print("Trying to call the response without setting initial values results in an error")
        simple_module.response()
    except TypeError as e:
        print("ERROR OBTAINED: \n TypeError: ", e.__str__())

    # To correctly do the forward calculation, a corect state need to be set in x1 and x2
    x1.state = 2.0  # Values for the for the forward calculation
    x2.state = 3.0
    print(f"\nState initialized to {x1.tag} = {x1.state}, {x2.tag} = {x2.state}")

    # Now execute response
    print("Call response")
    simple_module.response()

    # The state of the output signal can be accessed using <Signal>.state again
    print(f"The result: {y.tag} = {y.state}")

    print("\n-- MostSimple Module: Sensitivity analysis by back-propagation")
    # Calculate sensitivities
    print("\nIf no seed is given, no sensitivities will be calculated")
    simple_module.sensitivity()
    print(f"dg/d{x1.tag} = {x1.sensitivity}")
    print(f"dg/d{x2.tag} = {x2.sensitivity}")

    print("\nSeed dg/dy1 = 1.0, so we can calculate dy1/dx1 and dy1/dx2")
    # An initial 'seed' sensitivity of the response you're interested in needs to be set. We can do this by setting
    # the <Signal>.sensitivity property
    y.sensitivity = 1.0
    simple_module.sensitivity()
    # The sensitivities of the input signals can now be accessed by <Signal>.sensitivity
    print(f"dg/d{x1.tag} = {x1.sensitivity}")
    print(f"dg/d{x2.tag} = {x2.sensitivity}")

    print("\nWhen reset is not called after the sensitivity calculation, the results will not be correct.")
    print("Seed dg/dy1 = 1.0 again (not strictly necessary, since the value already was seeded)")
    y.sensitivity = 1.0
    simple_module.sensitivity()
    print(f"Incorrect sensitivity dy1/d{x1.tag} = {x1.sensitivity}")
    print(f"Incorrect sensitivity dy1/d{x2.tag} = {x2.sensitivity}")
    print("The values are now double of what they're supposed to be, because they're added to what we already had.")

    print("\nRESET! And seed dg/dy2 = 1.0 to calculate the other sensitivities")
    simple_module.reset()  # !! DON'T FORGET TO RESET, ELSE SENSITIVITIES FROM PREVIOUS RUNS WILL CONTAMINATE YOUR RESULT !!
    y.sensitivity = 1.0
    simple_module.sensitivity()
    print(f"dg/d{x1.tag} = {x1.sensitivity}")
    print(f"dg/d{x2.tag} = {x2.sensitivity}")

    # You can always check your module with finite differencing
    finite_difference(simple_module)
