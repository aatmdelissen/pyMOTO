""" Example 1: Simple scalar network
This example demonstrates how a simple Module is implemented. Different methods of data storage inside the Module are
shown, and the ordering of the Modules in the Network can be changed to obtain other behavior. Basic usage of Signals,
sensitivity calculation, and finite-difference validation is demonstrated.
"""
from pymoto import Module, Signal, Network, finite_difference
import math


# Module definitions
class ModuleA(Module):
    """ Evaluates y = x2 * sin(x1)
    In this module, the state variables are stored internally during response() for use in the sensitivity()
    """
    def _response(self, x1, x2):
        # Store state for use in sensitivity
        self.x1 = x1
        self.x2 = x2
        return self.x2 * math.sin(self.x1)

    def _sensitivity(self, df_dy):
        df_dx1 = df_dy * self.x2 * math.cos(self.x1)
        df_dx2 = df_dy * math.sin(self.x1)
        return df_dx1, df_dx2


class ModuleB(Module):
    """ Evaluates y = cos(x1) * cos(x2)
    The derivatives are already calculated during the response(), for easy use in sensitivity()
    """
    def _response(self, x1, x2):
        # Already calculate the state derivative
        self.dy_dx1 = math.sin(x1) * math.cos(x2)
        self.dy_dx2 = math.cos(x1) * math.sin(x2)
        return math.cos(x2) * math.cos(x1)

    def _sensitivity(self, df_dy):
        df_dx1 = - df_dy * self.dy_dx1
        df_dx2 = - df_dy * self.dy_dx2
        return df_dx1, df_dx2


class ModuleC(Module):
    """ Evaluates y = x1^2 * (1 + x2)
    Obtain the state variables during sensitivity()
    """
    def _response(self, x1, x2):
        return x1**2 * (1 + x2)

    def _sensitivity(self, df_dy):
        # Obtain input state from signals
        x1 = self.sig_in[0].state
        x2 = self.sig_in[1].state
        df_dx1 = 2 * df_dy * x1 * (1 + x2)
        df_dx2 = df_dy * x1 * x1
        return df_dx1, df_dx2


if __name__ == '__main__':
    print(__doc__)

    # --- SETUP ---
    # Declare the signals
    x = Signal('x')
    y = Signal('y')
    z = Signal('z')
    a = Signal('a')
    b = Signal('b')
    g = Signal('g')

    # Set initial values
    x.state = 1.0
    y.state = 0.8
    z.state = 3.4

    # Create the modules
    # Depending on how the input and output signals are routed between the modules, different behavior can be created
    ordering = 0
    if ordering == 0:
        # A __
        #     \
        #      --> C
        # B __/
        m1 = ModuleA([x, y], a)
        m2 = ModuleB([y, z], b)
        m3 = ModuleC([a, b], g)
    elif ordering == 1:
        # B __
        #     \
        #      --> A
        # C __/
        print("Using an alternative module order")
        m1 = ModuleB([x, y], a)
        m2 = ModuleC([y, z], b)
        m3 = ModuleA([a, b], g)
    else:
        m1, m2, m3 = None, None, None

    # Create a network of modules
    func = Network(m1, m2, m3)

    print("\nCurrent network:")
    print(" -> ".join([type(m).__name__ for m in func.mods]))

    # --- FORWARD ANALYSIS ---
    # Perform forward analysis
    func.response()

    print(f"\nThe response is  g(x={x.state}, y={y.state}, z={z.state}) = {g.state}")

    # --- BACKPROPAGATION ---
    # Clear previous sensitivities
    func.reset()

    # Seed the response sensitivity
    g.sensitivity = 1.0

    # Calculate the sensitivities
    func.sensitivity()

    print("\nThe sensitivities are:")
    print(f"d{g.tag}/d{x.tag} = {x.sensitivity}")
    print(f"d{g.tag}/d{y.tag} = {y.sensitivity}")
    print(f"d{g.tag}/d{z.tag} = {z.sensitivity}")

    # --- Finite difference checks ---
    # On the individual modules
    finite_difference(m1)

    finite_difference(m2)

    finite_difference(m3)

    # Do a finite difference check on the entire network
    finite_difference(func)
