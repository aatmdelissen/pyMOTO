""" Example: Module template for a generic module
It shows possibilities for a generic Module, and also the importance of the sensitivity reset.
"""
from pymoto import Module, Signal, finite_difference


class MyNewModule(Module):
    """ This is my module that does awesome stuff.

    The (Module) is required for your module to be a module, with correct behaviour.
    This example is a module with two inputs and two outputs.
    """
    def _prepare(self, my_argument, optional_arg='foo'):
        """
        This function is called by the initialization of the superclass. It can use the input parameters entered into
        the Module.create() method.
        """
        print('>> Prepare my module')
        print('\tmy_argument = {}'.format(my_argument))
        print('\toptional_argument = {}'.format(optional_arg))

    def _response(self, x1, x2):
        """
        This function calculates a response based on a multiple input values, here for example 2. Multiple outputs can
        easily be added. Also, different response behaviours can be implemented, based on the number of inputs (function
        overloading). The 'self' object can be used to save state variables.

        :param x1: First input variable
        :param x2: Second input variable
        :return: The results of the calculation
        """
        print('>> Do my response calculation')

        # Incorrect data
        if x1 is None or x2 is None:
            raise RuntimeError("You forgot to set {} and {}".format(self.sig_in[0].tag, self.sig_in[1].tag))

        # Store data, which might be needed for the sensitivity calculation
        self.x1 = x1
        self.x2 = x2

        # Calculate two response values
        v1 = x1 * x2
        v2 = x1 + x2

        # Return the results
        return v1, v2

    def _sensitivity(self, df_dv1, df_dv2):
        """
        This function calculate the (backward) sensitivity. It should handle None (zero sensitivity) as incoming adjoint
        variable. If both are None, the sensitivity will not be called.

        :param df_dv1: The adjoint variable of the first output
        :param df_dv2: The adjoint variable of the second output
        :return: The adjoint variables (sensitivities) of the inputs
        """
        print('>> Do my sensitivity calculation')

        # Calculate the gradients with chain-rule
        # First initialize sensitivities with the correct size containing all zeros
        df_dx1 = self.x1 * 0
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

    def _reset(self):
        """
        This function is used for reset, called whenever a new sensitivity is to be calculated, triggered by calling
        reset(). This can be used to reset any internal storage required for sensitivity calculation
        """
        print('>> Reset my module')


if __name__ == "__main__":
    print(__doc__)
    print("_" * 80)
    print("PART 1: Setup")

    # The print function lists all possible Module types
    print("Show all possible modules defined, it also lists the new locally defined module")
    Module.print_children()

    # Create signals for the inputs. The argument is the 'tag' of the signal, which is optional.
    # The signal's 'tag' can be seen as its name, which can be useful for printing or when
    x1 = Signal("x1")

    # Also create a second input signal (as our module has two inputs)
    x2 = Signal("x2")

    # And create two signals for the outputs
    y1 = Signal("y1")
    y2 = Signal("y2")

    # The module can be instantiated using the constructor. The first argument is a list of input signals, and the
    # second argument is a list of output arguments. The other (keyword) arguments are directly passed to the _prepare()
    # function of the module.
    print("\nInstantiate directly:")
    the_mod = MyNewModule([x1, x2], [y1, y2], 'my_arg_data', optional_arg=3.14)

    # It can also be instantiated using Module.create, by referencing its name (case insensitive) as first argument.
    # All the other arguments remain identical to before.
    print("\nInstantiate by Module.create:")
    also_the_mod = Module.create("MyNewModule", [x1, x2], [y1, y2], 'my_arg_data1', optional_arg='bar')

    print("_"*80)
    print("PART 2: Forward analysis")

    try:
        print("\nTrying to call the response without setting initial values results in an error")
        the_mod.response()
    except RuntimeError as e:
        print("ERROR OBTAINED: ", e.__str__())

    # Set the initial values to the state of the input signals using <Signal>.state
    x1.state = 2.0
    x2.state = 3.0

    print("\nState initialized to {0} = {1}, {2} = {3}".format(x1.tag, x1.state, x2.tag, x2.state))
    print("Call response")
    the_mod.response()
    # The states of the output signals can be accessed using <Signal>.state
    print("The results: {0} = {1}, {2} = {3}".format(y1.tag, y1.state, y2.tag, y2.state))

    # Sensitivity analysis
    print("_"*80)
    print("PART 3: Backpropagation (sensitivity analysis)")

    # Calculate sensitivities
    print("\nIf no seed is given, no sensitivities will be calculated")
    the_mod.sensitivity()
    print("dg/d{} = {}".format(x1.tag, x1.sensitivity))
    print("dg/d{} = {}".format(x2.tag, x2.sensitivity))

    print("\nSeed dg/dy1 = 1.0, so we can calculate dy1/dx1 and dy1/dx2")
    # An initial 'seed' sensitivity of the last response you're interested in needs to be set. We can do this by setting
    # the <Signal>.sensitivity property
    y1.sensitivity = 1.0
    the_mod.sensitivity()
    # The sensitivities of the initial signals can also be accessed by <Signal>.sensitivity
    print("dy1/d{} = {}".format(x1.tag, x1.sensitivity))
    print("dy1/d{} = {}".format(x2.tag, x2.sensitivity))

    print("\nWhen reset is not called after the sensitivity calculation, the results will not be correct.")
    print("Seed dg/dy1 = 1.0 again (not strictly necessary, since the value already was seeded)")
    y1.sensitivity = 1.0
    the_mod.sensitivity()
    print("Incorrect sensitivity dy1/d{} = {}".format(x1.tag, x1.sensitivity))
    print("Incorrect sensitivity dy1/d{} = {}".format(x2.tag, x2.sensitivity))
    print("The values are now double of what they're supposed to be, because they're added to what we already had.")

    print("\nRESET! And seed dg/dy2 = 1.0 to calculate the other sensitivities")
    the_mod.reset()  # !! DON'T FORGET TO RESET, ELSE SENSITIVITIES FROM PREVIOUS RUNS WILL CONTAMINATE YOUR RESULT !!
    y2.sensitivity = 1.0
    the_mod.sensitivity()
    print("dy2/d{} = {}".format(x1.tag, x1.sensitivity))
    print("dy2/d{} = {}".format(x2.tag, x2.sensitivity))

    # You can always check your module with finite differencing
    finite_difference(the_mod)
