""" Example 2: Generic scalar-to-scalar modules """
from pyModular import MathGeneral, Signal, Interconnection, finite_difference

#foo
if __name__ == '__main__':
    print("Example 2: Generic scalar-to-scalar-module (with MathGeneral)")

    # SETUP
    x = Signal('x')
    y = Signal('y')
    z = Signal('z')
    a = Signal('a')
    b = Signal('b')
    g = Signal('g')

    # Initial values
    x.set_state(1.0)
    y.set_state(0.8)
    z.set_state(3.4)

    ordering = 0
    if ordering == 0:
        # Create modules
        m1 = MathGeneral([x, y], a, "inp1 * sin(inp0)")  # Also "x" and "y" can be used instead of "inp0" and "inp1"
        m2 = MathGeneral([y, z], b, "cos(inp0) * cos(inp1)")
        m3 = MathGeneral([a, b], g, "inp0^2 * (1 + inp1)")
    elif ordering == 1:
        m1 = MathGeneral([x, y], a, "cos(inp0) * cos(inp1)")
        m2 = MathGeneral([y, z], b, "inp0^2 * (1 + inp1)")
        m3 = MathGeneral([a, b], g, "inp1 * sin(inp0)")
    else:
        m1, m2, m3 = None, None, None
        exit()

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

    # Do a finite difference check
    finite_difference(m1)

    finite_difference(m2)

    finite_difference(m3)

    finite_difference(func)
