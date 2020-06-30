""" Template for a generic module """
from pyModular import Module, Signal, finite_difference


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

        # Store data
        self.x1 = x1
        self.x2 = x2

        # Calculate response
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
        df_dx1 = self.x1 * 0
        df_dx2 = self.x2 * 0

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
    print("_" * 80)
    print("PART 1: Setup")

    # The print function lists all possible Module types
    print("Show all possible modules defined, it also lists the new locally defined module")
    Module.print_children()

    x1 = Signal("x1")
    x2 = Signal("x2")
    y1 = Signal("y1")
    y2 = Signal("y2")

    # The module can be instantiated using the constructor
    print("\nInstantiate directly:")
    the_mod = MyNewModule([x1, x2], [y1, y2], 'my_arg_data', optional_arg=3.14)

    # It can also be instantiated using Module.create, by referencing its name (case insensitive):
    print("\nInstantiate by Module.create:")
    also_the_mod = Module.create("MyNewModule", [x1, x2], [y1, y2], 'my_arg_data1', optional_arg='bar')

    print("_"*80)
    print("PART 2: Forward analysis")

    try:
        # When no values are set into the signal's state, we get an error
        print("\nTry calling the response without initial values results in an error")
        the_mod.response()
    except TypeError as e:
        print("ERROR OBTAINED: ", e.__str__())

    # Set the initial values
    x1.set_state(2.0)
    x2.set_state(3.0)

    print("\nState initialized to {0} = {1}, {2} = {3}".format(x1.tag, x1.get_state(), x2.tag, x2.get_state()))
    print("Call response")
    the_mod.response()
    print("The results: {0} = {1}, {2} = {3}".format(y1.tag, y1.get_state(), y2.tag, y2.get_state()))

    print("_"*80)
    print("PART 3: Backpropagation (sensitivity analysis)")

    # Calculate sensitivities
    print("\nIf no seed is given, no sensitivities will be calculated")
    the_mod.sensitivity()
    print("dg/d{} = {}".format(x1.tag, x1.get_sens()))
    print("dg/d{} = {}".format(x2.tag, x2.get_sens()))

    print("\nSeed dg/dy1 = 1.0, so we can calculate dy1/dx1 and dy1/dx2")
    y1.set_sens(1.0)
    the_mod.sensitivity()
    print("dy1/d{} = {}".format(x1.tag, x1.get_sens()))
    print("dy1/d{} = {}".format(x2.tag, x2.get_sens()))

    print("\nRESET! And seed dg/dy2 = 1.0 to calculate the other sensitivities")
    the_mod.reset()  # DON'T FORGET THIS RESET, ELSE SENSITIVITIES FROM PREVIOUS RUNS WILL CONTAMINATE YOUR RESULT!
    y2.set_sens(1.0)
    the_mod.sensitivity()
    print("dy2/d{} = {}".format(x1.tag, x1.get_sens()))
    print("dy2/d{} = {}".format(x2.tag, x2.get_sens()))

    # You can always check your module with finite differencing
    finite_difference(the_mod)
