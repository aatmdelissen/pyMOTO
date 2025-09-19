"""Custom: Creating a simple module
===================================

How to create a simple custom module

In this example it is shown how you can create your own basic module in `pymoto` that multiplies two values. Both the 
response (forward) and sensitivity (backward) are implemented and are tested with the 
:py:func:`pymoto.finite_difference` function.
"""
import pymoto as pym


class MyModule(pym.Module):
    """ Example of a simple module with two inputs and one output """
    def __call__(self, x1, x2):
        """ Forward-path calculation is put here """
        print(f'[{type(self).__name__}] Do my response calculation')
        print(f'\tInputs are {x1} and {x2}')
        return x1 * x2

    def _sensitivity(self, df_dy):
        """ Backward-path sensitivity calculation here.
        In case df_dy is None, the function will automatically not be called
        """
        print(f'[{type(self).__name__}] Do my sensitivity calculation')
        x1, x2 = self.get_input_states()  # Get the input states if required
        df_dx1 = df_dy * x2  # Apply chain rule df/dx1 = df/dy * dy/dx1
        df_dx2 = df_dy * x1
        return df_dx1, df_dx2  # Return the sensitivities with respect to input signals


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
    my_module = MyModule()
    
    # After the module is created, it is connected to the input signals. 
    # It will be run once and an output signal is created.
    print("\n-- Connect module and run forward analysis:")
    y = my_module(x1, x2)
    y.tag = 'y'  # Set a name for the output signal

    # The state of the output signal can be accessed using its `state`
    print(f"The result: {y.tag} = {y.state}")

    # Calculate sensitivities
    print("\n-- Sensitivity analysis by back-propagation")
    print("\nSeed dy/dy = 1.0, so we can calculate dy/dx1 and dy/dx2")
    # An initial 'seed' sensitivity of the response you're interested in needs to be set. We can do this by setting
    # the `sensitivity` property
    y.sensitivity = 1.0
    my_module.sensitivity()  # Run backpropagation to calculate sensitivities
    # The sensitivities of the input signals can now be accessed from the `sensitivity` property
    print(f"dy/d{x1.tag} = {x1.sensitivity}")
    print(f"dy/d{x2.tag} = {x2.sensitivity}")

    # You can always check your module with finite differencing
    pym.finite_difference([x1, x2], y, random=False)
