[![10.5281/zenodo.7708738](https://zenodo.org/badge/DOI/10.5281/zenodo.7708738.svg)](https://doi.org/10.5281/zenodo.7708738) 
[![anaconda.org/aatmdelissen/pymoto](https://anaconda.org/aatmdelissen/pymoto/badges/version.svg)](https://anaconda.org/aatmdelissen/pymoto)
[![pypi.org/project/pyMOTO](https://badge.fury.io/py/pyMOTO.svg)](https://pypi.org/project/pyMOTO/)

# pyMOTO

* [Link to Documentation](https://pymoto.readthedocs.io)
* [Link to GitHub](https://github.com/aatmdelissen/pyMOTO)

Modular topology optimization framework with semi-automatic derivatives. The two main types `Module` and `Signal`
are used to implement a problem formulation to be optimized. The `Module` implements functionality (and design 
sensitivity calculations) and `Signal` carries data of both the variables and their derivatives. 

Sensitivity calculations are done based on backpropagation. The final value of interest is seeded with sensitivity
value $\frac{\textup{d}f}{\textup{d}f}=1$. Then the modules are executed in reverse order, each applying the chain rule.
As example for a `Module` which maps $x\rightarrow y$, only the following operation needs to be implemented:

$$
\frac{\textup{d}f}{\textup{d}x} = \frac{\textup{d}f}{\textup{d}y}\frac{\textup{d}y}{\textup{d}x}\text{.} 
$$

In this way, the order of modules can easily be adapted without having to worry about sensitivities, as these are 
automatically calculated.

# Quick start installation
1. Make sure you have Python running in some kind of virtual environment (e.g. 
[conda](https://docs.conda.io/projects/conda/en/stable/), [miniconda](https://docs.conda.io/en/latest/miniconda.html),
[venv](https://realpython.com/python-virtual-environments-a-primer/))
2. Install the pymoto Python package (and its dependencies)
   - Option A (conda): If you are working with Conda, install by `conda install -c aatmdelissen pymoto`
   - Option B (pip): Type `pip install pymoto` into your console to install
3. Download one of the examples found in the repository's example folder 
([here](https://github.com/aatmdelissen/pyMOTO/tree/master/examples))
4. Run the example by typing `python ex_....py` in the console


A local installation for development in `pyMOTO` can be done by first downloading the entire git repo, and then calling 
`pip install -e .` in the `pyMOTO` folder (of course from within your virtual environment).

## Dependencies
* **NumPy** - Dense linear algebra and solvers
* **SciPy** - Sparse linear algebra and solvers
* **SymPy** - Symbolic differentiation for `MathGeneral` module
* **Matplotlib** - Plotting and visualisation
* (optional) **SAO** - Sequential approximated optimizers
* (optional) [**opt_einsum**](https://optimized-einsum.readthedocs.io/en/stable/install.html) - Optimized function for `EinSum` module

For fast linear solvers for sparse matrices:
* (optional) [**pypardiso**](https://github.com/haasad/PyPardisoProject) - Uses the Intel OneAPI PARDISO solver (recommended)
* (optional) [**scikit-umfpack**](https://scikit-umfpack.github.io/scikit-umfpack/install.html) - Fast LU linear solver based on UMFPACK
* (optional) [**scikit-sparse**](https://github.com/scikit-sparse/scikit-sparse) - Fast Cholesky solver based on CHOLMOD
* (optional) [**cvxopt**](https://cvxopt.org/install/index.html) - Another fast Cholesky solver based on CHOLMOD

__Note on linear solvers for sparse matrices:__ Scipy implements a version of LU which is quite slow. To increase the 
speed of the optimization, `pypardiso` is recommended as it contains a very robust and flexible solver for symmetric 
and asymmetric matrices. An alternative is `scikit-umfpack` which provides a fast LU factorization. For symmetric 
matrices a Cholesky factorization is recommended (not provided with Scipy), which can be used by either installing 
`scikit-sparse` or `cvxopt`.

# License
pyMOTO is available under te [MIT License](https://opensource.org/licenses/MIT).
