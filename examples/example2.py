""" Example 2: Expressions
Identical to example 1, but by using MathGeneral module. This demonstrates the possiblity for a user to enter a formula
as a string. Next to this, it is possible to use this module for other data-types than scalars, such as vectors.
"""
from pymoto import MathGeneral, Signal, Network, finite_difference

if __name__ == '__main__':
    print(__doc__)

    # --- SETUP ---
    x = Signal('x')
    y = Signal('y')
    z = Signal('z')
    a = Signal('a')
    b = Signal('b')
    g = Signal('g')

    # Initial values
    use_vector = True
    if use_vector:
        print("The initial values are vectors")
        import numpy as np
        x.state = np.array([1.0, 2.0, 3.0, 4.0])
        y.state = np.array([0.8, 1.2, 1.8, 2.6])
        z.state = np.array([3.4, 8.5, 4.1, 6.3])
    else:
        print("The initial values are scalar")
        x.state = 1.0
        y.state = 0.8
        z.state = 3.4

    # Create modules
    ordering = 0
    if ordering == 0:
        m1 = MathGeneral([x, y], a, "x * sin(y)")  # Also "inp0" and "inp1" can be used instead of "x" and "y"
        m2 = MathGeneral([y, z], b, "cos(inp0) * cos(inp1)")
        m3 = MathGeneral([a, b], g, "inp0^2 * (1 + inp1)")
    elif ordering == 1:
        print("Using an alternative module order")
        m1 = MathGeneral([x, y], a, "cos(inp0) * cos(inp1)")
        m2 = MathGeneral([y, z], b, "inp0^2 * (1 + inp1)")
        m3 = MathGeneral([a, b], g, "inp1 * sin(inp0)")
    else:
        m1, m2, m3 = None, None, None
        exit()

    # Create interconnection
    func = Network(m1, m2, m3)

    print("\nCurrent interconnection:")
    print(" -> ".join([type(m).__name__ for m in func.mods]))

    # --- FORWARD ANALYSIS ---
    # Perform forward analysis
    func.response()

    print("\nThe response is  g(x={0}, y={1}, z={2}) = {3}".format(x.state, y.state, z.state, g.state))

    # --- BACKPROPAGATION ---
    # Clear previous sensitivities
    func.reset()

    # Seed the response sensitivity
    g.sensitivity = 1.0

    # Calculate the sensitivities
    func.sensitivity()

    print("\nThe sensitivities are:")
    print("d{0}/d{1} = {2}".format(g.tag, x.tag, x.sensitivity))
    print("d{0}/d{1} = {2}".format(g.tag, y.tag, y.sensitivity))
    print("d{0}/d{1} = {2}".format(g.tag, z.tag, z.sensitivity))

    # --- Finite difference checks ---
    # On the individual modules
    finite_difference(m1)

    finite_difference(m2)

    finite_difference(m3)

    # Do a finite difference check on the entire network
    finite_difference(func)
