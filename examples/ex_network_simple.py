"""Custom: Simple scalar network
================================

Creating a network with multiple custom modules

This example demonstrates how multiple simple modules are implemented. Different methods of data storage inside the 
odule are shown. Next to that, the effect of ordering the modules in a :py:class:`pymoto.Network` is demonstrated. This 
allows for different mathematical behavior, while keeping the same implementation of modules: no additional effort needs
to be made to keep the sensitivities consistent. Also basic usage of :py:class:`pymoto.Signal`, response and sensitivity
calculation, and validation with :py:func:`pymoto.finite_difference` is demonstrated.

This example is identical in behavior to :ref:`sphx_glr_auto_examples_ex_network_simple_mathexpression.py`, but uses 
manually implemented sensitivities instead of automatically generated ones.
"""
import pymoto as pym
import math


# Module definitions
class ModuleA(pym.Module):
    """ Evaluates y = x2 * sin(x1)
    In this module, the state variables are stored internally during response() for use in the sensitivity()
    """
    def __call__(self, x1, x2):
        # Store state for use in sensitivity
        self.x1 = x1
        self.x2 = x2
        return self.x2 * math.sin(self.x1)

    def _sensitivity(self, df_dy):
        df_dx1 = df_dy * self.x2 * math.cos(self.x1)
        df_dx2 = df_dy * math.sin(self.x1)
        return df_dx1, df_dx2


class ModuleB(pym.Module):
    """ Evaluates y = cos(x1) * cos(x2)
    The derivatives are already calculated during the response(), for easy use in sensitivity()
    """
    def __call__(self, x1, x2):
        # Already calculate the state derivative
        self.dy_dx1 = math.sin(x1) * math.cos(x2)
        self.dy_dx2 = math.cos(x1) * math.sin(x2)
        return math.cos(x2) * math.cos(x1)

    def _sensitivity(self, df_dy):
        df_dx1 = - df_dy * self.dy_dx1
        df_dx2 = - df_dy * self.dy_dx2
        return df_dx1, df_dx2


class ModuleC(pym.Module):
    """ Evaluates y = x1^2 * (1 + x2)
    Obtain the state variables during sensitivity()
    """
    def __call__(self, x1, x2):
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
    # Declare the signals and set initial values
    x = pym.Signal('x', 1.0)
    y = pym.Signal('y', 0.8)
    z = pym.Signal('z', 3.4)

    # Start building a network of modules
    with pym.Network() as fn:
        # Create the modules here
        # Depending on how the input and output signals are routed between the modules, different behavior can be
        # implemented

        ordering = 0  # Change this to 1 to see a different ordering of the modules
        if ordering == 0:
            # A __
            #     \
            #      --> C
            # B __/
            a = ModuleA()(x, y)
            b = ModuleB()(y, z)
            g = ModuleC()(a, b)
            a.tag, b.tag, g.tag = 'a', 'b', 'g'  # Set tags for the signals
        elif ordering == 1:
            # B __
            #     \
            #      --> A
            # C __/
            print("Using an alternative module order")
            a = ModuleB()(x, y)
            b = ModuleC()(y, z)
            g = ModuleA()(a, b)
            a.tag, b.tag, g.tag = 'a', 'b', 'g'  # Set tags for the signals
        


    print("\nCurrent network:")
    print(" -> ".join([type(m).__name__ for m in fn.mods]))

    print(f"The response is  g(x={x.state}, y={y.state}, z={z.state}) = {g.state}")
    
    # --- FORWARD ANALYSIS ---
    # Perform an extra forward analysis
    # Change the values of the input state
    x.state *= 2
    y.state += 0.1
    z.state -= 0.2

    fn.response()  # Run the forward analysis again
    print(f"The updated response is  g(x={x.state}, y={y.state}, z={z.state}) = {g.state}")

    # --- BACKPROPAGATION ---
    # Clear previous sensitivities (in this case it is redundant as the network is just created, but good practice)
    fn.reset()

    # Seed the response sensitivity
    g.sensitivity = 1.0

    # Calculate the sensitivities
    fn.sensitivity()

    print("\nThe sensitivities are:")
    print(f"d{g.tag}/d{x.tag} = {x.sensitivity}")
    print(f"d{g.tag}/d{y.tag} = {y.sensitivity}")
    print(f"d{g.tag}/d{z.tag} = {z.sensitivity}")

    # --- Finite difference checks ---
    # On the individual modules
    pym.finite_difference([x, y], a, random=False)  # From (x, y) to a
    pym.finite_difference([y, z], b, random=False)  # From (y, z) to b
    pym.finite_difference([a, b], g, random=False)  # From (a, b) to g

    # Do a finite difference check on the entire network
    pym.finite_difference([x, y, z], g, random=False)
