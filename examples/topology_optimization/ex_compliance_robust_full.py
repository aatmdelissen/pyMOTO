r"""Full robust formulation
===========================

Robust formulation, where each of the eroded, nominal, and dilated designs are evaluated.

This is the 'expensive' version of the robust formulation for compliance problems. The poor man's version can be found 
in :ref:`sphx_glr_auto_examples_topology_optimization_ex_compliance_robust.py`, which only evaluates the eroded version.

In this particular example, Python functions are used to generate the repeated sequence of modules used for geometry 
projection and performance evaluation of each design.

References:
  Wang, F., Lazarov, B. S., & Sigmund, O. (2011).
  On projection methods, convergence and robust formulations in topology optimization.
  Structural and multidisciplinary optimization, 43, 767-784.
  DOI: https://doi.org/10.1007/s00158-010-0602-y
"""
import numpy as np
import pymoto as pym


class Continuation(pym.Module):
    """ Module that generates a continuated value """

    def __init__(self, start=0.0, stop=1.0, nsteps=80, stepstart=10, interval=10):
        self.startval = start
        self.endval = stop
        self.interval = interval
        self.dval = (stop - start) / nsteps
        self.nstart = stepstart
        self.iter = -1
        self.val = self.startval

    def __call__(self):
        maxval = max(self.startval, self.endval)
        minval = min(self.startval, self.endval)
        if self.iter % self.interval == 0:  # Only update value every `interval` iterations
            self.val = np.clip(self.startval + self.dval * (self.iter - self.nstart), minval, maxval)
        
        self.iter += 1
        return self.val


if __name__ == '__main__':
    print(__doc__)

    # Define the network as a normal python function
    def my_function(sx_in, s_beta, f, domain, eta=0.5, xmin=1e-9, bc=None):
        """Generate Heaviside projected eigenvalue calculation
        """
        # Heaviside projection
        heaviside = "(tanh(inp1 * {0}) + tanh(inp1 * (inp0 - {0}))) / (tanh(inp1 * {0}) + tanh(inp1 * (1 - {0})))"
        sx_projected = pym.MathExpression(heaviside.format(eta))(sx_in, s_beta)

        # SIMP material interpolation
        sx_SIMP = pym.MathExpression(f"{xmin} + {1-xmin}*inp0^3")(sx_projected)

        # Stiffness matrix assembly
        sK = pym.AssembleStiffness(domain, bc=bc)(sx_SIMP)

        # Eigenvalue calculation
        s_u = pym.LinSolve()(sK, f)

        # Compliance calculation
        s_compl = pym.EinSum('i,i->')(s_u, f)

        return s_compl, sx_projected

    # Now that the function is defined, it can be used as a 'recipe' for various inputs. The function will add the 
    # modules that are requested and return the signal for 'frequency' in this case.

    # Setup a domain
    domain_2d = pym.VoxelDomain(100, 50)
    bc = domain_2d.get_dofnumber(domain_2d.nodes[0, :], ndof=2)
    f = np.zeros(domain_2d.nnodes*2)
    f[domain_2d.nodes[-1, 10]*2 + 1] = 1

    # Setup design vector
    sx = pym.Signal('x', np.ones(domain_2d.nel) * 0.5)
    sxfilt = pym.DensityFilter(domain_2d, radius=2)(sx)

    # Continuation parameter `beta`
    sbeta = Continuation(start=1e-3, stop=40.0, stepstart=10, nsteps=80, interval=5)()
    sbeta.tag = "beta"

    # Generate the three designs and evaluation for those
    s_compl_ero, sx_ero = my_function(sxfilt, sbeta, f, domain_2d, eta=0.7, bc=bc)
    s_compl_nom, sx_nom = my_function(sxfilt, sbeta, f, domain_2d, eta=0.5, bc=bc)
    s_compl_dil, sx_dil = my_function(sxfilt, sbeta, f, domain_2d, eta=0.3, bc=bc)

    # After this you can mix and match any of the outputs of the functions, to use for an optimization. 
    
    # Plot the nominal domain
    pym.PlotDomain(domain_2d)(sx_nom)

    # In this example the root mean square value of the compliances is minimized.
    s_compl_avg = pym.MathExpression("sqrt(inp0^2 + inp1^2 + inp2^2)")(s_compl_ero, s_compl_nom, s_compl_dil)
    s_obj = pym.Scaling(scaling=100.0)(s_compl_avg)

    # Perform the optimization
    pym.minimize_oc(sx, s_obj, maxit=100)

    # You can also evaluate the compliance for a range of eta values using the same function again
    eta_array = np.linspace(0.2, 0.8, 30)
    compliance_array = np.zeros(eta_array.size)
    for i, eta in enumerate(eta_array):
        s_compl, _ = my_function(sxfilt, sbeta, f, domain_2d, eta=eta, bc=bc)
        compliance_array[i] = s_compl.state

    # Show the compliance vs eta parameter. From this plot can clearly be seen that high eta (dilation) is always worse 
    # for performance in the case of compliance optimization. For other types of optimization (e.g. eigenfrequency) this
    # is not the case. Try it out for yourself if you can see this behavior by changing the function to do an 
    # eigensolve to calculate eigenfrequencies.
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(eta_array, compliance_array)
    plt.xlabel('eta')
    plt.ylabel('Compliance')
    plt.show()



