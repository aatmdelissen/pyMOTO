""" Example 1: Multiple scalar-to-scalar modules """
from pyModular import Module, Signal, Interconnection, finite_difference
from math import *


class ModuleA(Module):
    """ y = x2 * sin(x1) """
    def _response(self, x1, x2):
        # Store state for use in sensitivity
        self.x1 = x1
        self.x2 = x2
        return self.x2 * sin(self.x1)

    def _sensitivity(self, df_dy):
        df_dx1 = df_dy * self.x2 * cos(self.x1)
        df_dx2 = df_dy * sin(self.x1)
        return df_dx1, df_dx2


class ModuleB(Module):
    """ y = cos(x1) * cos(x2) """
    def _response(self, x1, x2):
        # Already calculate the state derivative
        self.dy_dx1 = sin(x1) * cos(x2)
        self.dy_dx2 = cos(x1) * sin(x2)
        return cos(x2) * cos(x1)

    def _sensitivity(self, df_dy):
        df_dx1 = - df_dy * self.dy_dx1
        df_dx2 = - df_dy * self.dy_dx2
        return df_dx1, df_dx2


class ModuleC(Module):
    """ y = x1^2 * (1 + x2) """
    def _response(self, x1, x2):
        return x1**2 * (1 + x2)

    def _sensitivity(self, df_dy):
        # Obtain input state from signals
        x1 = self.sig_in[0].get_state()
        x2 = self.sig_in[1].get_state()
        df_dx1 = 2 * df_dy * x1 * (1 + x2)
        df_dx2 = df_dy * x1 * x1
        return df_dx1, df_dx2


if __name__ == '__main__':
    print("Example 1: Multiple scalar-to-scalar modules")

    # Declare the signals
    x = Signal('x')
    y = Signal('y')
    z = Signal('z')
    a = Signal('a')
    b = Signal('b')
    g = Signal('g')

    # Set initial values
    x.set_state(1.0)
    y.set_state(0.8)
    z.set_state(3.4)

    ordering = 0
    if ordering == 0:
        # Create modules
        m1 = ModuleA([x, y], a)
        m2 = ModuleB([y, z], b)
        m3 = ModuleC([a, b], g)
    elif ordering == 1:
        m1 = ModuleB([x, y], a)
        m2 = ModuleC([y, z], b)
        m3 = ModuleA([a, b], g)
    else:
        m1, m2, m3 = None, None, None

    # Create interconnection
    func = Interconnection(m1, m2, m3)

    print("Current interconnection:")
    print(" -> ".join([type(m).__name__ for m in func.mods]))

    # FORWARD ANALYSIS
    # Perform forward analysis
    func.response()

    print("The response is  g(x={0}, y={1}, z={2}) = {3}".format(x.get_state(), y.get_state(),
                                                                 z.get_state(), g.get_state()))
    # BACKPROPAGATION
    # Clear previous sensitivities
    func.reset()

    # Seed the response sensitivity
    g.set_sens(1.0)

    # Calculate the sensitivities
    func.sensitivity()

    print("The sensitivities are:")
    print("d{0}/d{1} = {2}".format(g.tag, x.tag, x.get_sens()))
    print("d{0}/d{1} = {2}".format(g.tag, y.tag, y.get_sens()))
    print("d{0}/d{1} = {2}".format(g.tag, z.tag, z.get_sens()))

    # Finite difference checks
    # On ModuleA
    finite_difference(m1)

    finite_difference(m2)

    finite_difference(m3)

    # Do a finite difference check on the entire system
    finite_difference(func)
