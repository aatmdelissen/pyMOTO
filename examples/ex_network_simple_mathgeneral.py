""" 
Expressions in MathGeneral
==========================

Identical to :ref:`sphx_glr_ex_network_simple.py`, but by using MathGeneral module. This demonstrates the possiblity for 
a user to enter a formula as a string. Next to this, it is possible to use this module for other data-types than 
scalars, such as vectors.
"""
import pymoto as pym

if __name__ == '__main__':
    print(__doc__)

    # --- SETUP ---
    x = pym.Signal('x')
    y = pym.Signal('y')
    z = pym.Signal('z')

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
    with pym.Network() as func:
        ordering = 0
        if ordering == 0:
            a = pym.MathGeneral("inp0 * sin(inp1)")(x, y)
            # Instad of  'inp0' and 'inp1', also the tags ('y', 'z') can be used (if they are defined!)
            b = pym.MathGeneral("cos(y) * cos(z)")(y, z) 
            g = pym.MathGeneral("inp0^2 * (1 + inp1)")(a, b)  # Signals a and b do not have a tag, so inp0 and inp1 are used
        elif ordering == 1:
            print("Using an alternative module order")
            a = pym.MathGeneral("cos(inp0) * cos(inp1)")(x, y)
            b = pym.MathGeneral("inp0^2 * (1 + inp1)")(y, z) 
            g = pym.MathGeneral("inp1 * sin(inp0)")(a, b)
        else:
            raise RuntimeError("Unknown ordering")

        if use_vector:
            # In case of vector, convert to scalar by summing over the vector elements
            g_vector = g
            g = pym.EinSum('i->')(g_vector)  
        g.tag = 'g'

    print("\nCurrent Network:")
    print(" -> ".join([type(m).__name__ for m in func.mods]))

    print(f"\nThe result is  g(x={x.state}, y={y.state}, z={z.state}) = {g.state}")
    
    # --- FORWARD ANALYSIS ---
    # Perform forward analysis with updated values
    x.state *= 2
    y.state += 0.1
    func.response()
    print(f"\nThe updated result is  g(x={x.state}, y={y.state}, z={z.state}) = {g.state}")
    
    # --- BACKPROPAGATION ---
    # Seed the response sensitivity
    g.sensitivity = 1.0

    # Calculate the sensitivities
    func.sensitivity()

    print("\nThe sensitivities are:")
    print("d{0}/d{1} = {2}".format(g.tag, x.tag, x.sensitivity))
    print("d{0}/d{1} = {2}".format(g.tag, y.tag, y.sensitivity))
    print("d{0}/d{1} = {2}".format(g.tag, z.tag, z.sensitivity))

    # --- Finite difference checks ---
    # Do a finite difference check on the entire network
    pym.finite_difference(x, g, function=func, random=False)
