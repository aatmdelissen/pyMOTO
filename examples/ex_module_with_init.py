""" 
Creating module with initialization
===================================

This examples demonstrates how to make a module with an ininitialization function. Constant values and other parameters 
can be passed here. 
"""
import pymoto as pym


class ModuleWithInit(pym.Module):
    """ This module takes a parameter during initialization
    Example usage:
    >>> import pymoto as pym
    >>> x = pym.Signal('x', state=3.5)
    >>> y = pym.Signal('y')
    >>> y = WithInit(1.2, optional_value='bar')(x, y)  # doctest: +ELLIPSIS
    [WithInit] Prepare my module
    value = 1.2
    optional_value = bar
    [WithInit] Do my response calculation
    Message = bar, x = 3.5, y=4.2
    ...
    >>> y.state
    4.2
    """
    def __init__(self, value, optional_value='foo'):
        """ This prepare is called during initialization of the module, and can be used for set-up """
        print(f'[{type(self).__name__}] Initialize my module')
        print(f'value = {value}')
        print(f'optional_value = {optional_value}')
        self.value = value
        self.optional_value = optional_value

    def __call__(self, x):
        print(f'[{type(self).__name__}] Do my response calculation')
        y = x * self.value
        print(f"Message = {self.optional_value}, x = {x}, y={y}")
        return y

    def _sensitivity(self, df_dy):
        print(f'[{type(self).__name__}] Do my sensitivity calculation')
        return df_dy * self.value


if __name__ == "__main__":
    print(__doc__)
    print("_" * 80)
    print("-- Module setup")

    # Create signals for the inputs. The argument is the 'tag' of the signal, which is optional.
    # The tag of the signal can be seen as its name, which can be useful for printing and debugging
    x = pym.Signal("x", 2.0)

    print(f"\nState initialized to {x.tag} = {x.state}")

    # The module is instantiated using the constructor. The values used in the `__init__` function can be passed here.
    print("Create Module:")
    my_module = ModuleWithInit(3.14, optional_value='bar')   # Module with extra (constant) parameters

    # After the module is created, it is connected to the input signal. 
    # It will be run once and an output signal is created.
    print("\n-- Connect module and run forward analysis:")
    y = my_module(x)
    y.tag = 'y'  # Set a name for the output signal

    # The state of the output signal can be accessed using `state` again
    print(f"The result: {y.tag} = {y.state}")

    print("\n-- Sensitivity analysis by back-propagation")
    # Calculate sensitivities
    print("\nSeed dy/dy = 1.0, so we can calculate dy/dx")
    # An initial 'seed' sensitivity of the response you're interested in needs to be set. We can do this by setting
    # the `sensitivity` property
    y.sensitivity = 1.0
    my_module.sensitivity()
    # The sensitivities of the input signals can now be accessed by <Signal>.sensitivity
    print(f"dy/d{x.tag} = {x.sensitivity}")

    # You can always check your module with finite differencing
    pym.finite_difference([x], y, random=False)
