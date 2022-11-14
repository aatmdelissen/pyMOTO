![logo](M_logo_256.png)

Modular topology optimization framework with semi-automatic derivatives. The two main types `Module` and `Signal`
are used to implement a problem formulation to be optimized. The `Module` implements functionality (and design 
sensitivity calculations) and `Signal` carries data of both the variables and their derivatives. 

Sensitivity calculations are done based on backpropagation. The final value of interest is seeded with sensitivity
value $\frac{\textup{d}f}{\textup{d}f}=1$. Then the modules are executed in reverse order, each applying the chain rule.
As example for a `Module` which maps $x\rightarrow y$, only the following operation needs to be implemented:
$$\frac{\textup{d}f}{\textup{d}x} = \frac{\textup{d}f}{\textup{d}y}\frac{\textup{d}y}{\textup{d}x}\text{.} $$
In this way, the order of modules can easily be adapted without having to worry about sensitivities, as these are 
automatically calculated.

# Installation
Install by calling `pip install -e .` in the `pyModular` folder (within your virtual environment).

## Dependencies
* **NumPy** - Dense linear algebra and solvers
* **SciPy** - Sparse linear algebra and solvers
* **SymPy** - Symbolic differentiation for `MathGeneral` module
* **Matplotlib** - Plotting and visualisation
* (optional) **SAO** - Sequential approximated optimizers
* (optional) **opt_einsum** - Optimized function for `EinSum` module
* (optional) **scikit-umfpack** - Fast LU linear solver based on UMFPACK
* (optional) **sksparse** - Fast Cholesky solver based on CHOLMOD
* (optional) **CVXopt** - Another fast Cholesky solver based on CHOLMOD
* (optional) **Intel OneAPI** - Non-python library with a fast PARDISO solver

__Note on linear solvers for sparse matrices:__ Scipy implements a version of LU which is quite slow. To increase the 
speed of the optimization, `Intel OneAPI` is recommended as it contains a very robust and flexible solver for symmetric 
and asymmetric matrices. An alternative is `scikit-umfpack` which provides a fast LU factorization. For symmetric 
matrices a Cholesky factorization is recommended (not provided with Scipy), which can be used by either installing 
`sksparse` or `cvxopt`.


## How to make Python fast with Intel OneAPI
Intel provides a toolkit with many fast math operations and solvers called OneAPI (basekit). 
It can easily be installed on Linux by for instance following the steps described in https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html
For other OSes installation can be found in https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html

The nice thing about OneAPI is that it also includes an optimized version of Python. To use it follow the next steps (Linux)

1. `source <intel install location>/intel/oneapi/setvars.sh` (usually installed in `/opt/intel` or `/opt/ud/intel`). This loads the Intel OneAPI package.
2. `conda create --name <venv_name> --clone base` to create a new conda virtual environment to work in.
3. `conda activate <venv_name>` to activate the virtual environment.

### Usage of multi-thread linear solvers
Intel has a Pardiso type linear solver for fast solution of large systems.
To use it.....

# License
PyModular is available under te [MIT License](https://opensource.org/licenses/MIT).
# aatmdelissen.github.io
