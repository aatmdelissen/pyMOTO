# pyMOTO | Modular Topology Optimization in Python

[![10.5281/zenodo.8138859](https://zenodo.org/badge/DOI/10.5281/zenodo.8138859.svg)](https://doi.org/10.5281/zenodo.8138859) 
[![anaconda.org/aatmdelissen/pymoto](https://anaconda.org/aatmdelissen/pymoto/badges/version.svg)](https://anaconda.org/aatmdelissen/pymoto)
[![pypi.org/project/pyMOTO](https://badge.fury.io/py/pyMOTO.svg)](https://pypi.org/project/pyMOTO/)

- **Documentation** [https://pymoto.readthedocs.io](https://pymoto.readthedocs.io)
- **Examples gallery** [https://pymoto.readthedocs.io/en/latest/auto_examples/index.html](https://pymoto.readthedocs.io/en/latest/auto_examples/index.html)
- **Github repository** [https://github.com/aatmdelissen/pyMOTO](https://github.com/aatmdelissen/pyMOTO)

![3D Thermal heat-sink topology optimization](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXVsN21weW16Z3JudHVleWFyOG1qYjgxcXB5eWdraGczZDlmZWJkdCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kaoKxLKQCE4Nomj5og/giphy.gif)

This python library offers modular and easily reconfigurable code to perform topology optimization. 
Already many ingredients and variations of topology optimization are implemented! Examples are:
- Density filtering, robust formulations
- 2D and 3D topology optimization
- Static and dynamic structural mechanics
- Compliant mechanisms
- Thermal and thermo-mechanic coupling
- Stress constraints
- Multigrid preconditioning with conjugate-gradient solver
- Various optimizers suited for topology optimization, such as Optimality Criteria (OC), Method of Moving Asymptotes (MMA), and its Globally Convergent version (GCMMA)
- [And more... ](https://pymoto.readthedocs.io/en/latest/auto_examples/index.html)

![3D Mechanical MBB beam topology optimization](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXMwMnl4Z3lzODF1bGp0ejY0M3lvYnNsY2FxOTVtc3ZxdGF6dWpvNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WB1xOQOQvT5zUqRlq5/giphy.gif)

In `pymoto` a topology optimization problem is broken down into reusable sub-components (called *Modules*), such as density filter, finite-element assembly, linear solve, linear algebra, etc. Next to behaving like a function would (*e.g.* filtering the density field), a `Module` also implements design sensitivities (*i.e.* derivatives or gradients with respect to the inputs of that `Module`) of that operation. These are required for efficiently solving the topology optimization problem, but are usually very cumbersome to implement on the *whole* optimization problem. With `pymoto` however, the partial sensitivities are already implemented in each `Module`. When modules are linked together in `pymoto`, the chain rule is used to calculate the total sensitivities of the optimization problem (backpropagation). In essence it is a semi-automatic way of calculating design sensitivities.

Due to the modularity of the framework, existing modules can be reused and connected in different configurations without having to worry about sensitivity correctness. This allows for great flexibility in rearranging the modules, enabling a whole range of topology optimization problems even with a limited set of modules. If any functionality is not supported in a `Module` within the default `pymoto` library, a user can easily extend the framework by creating custom modules.

# Quick start installation
1. Make sure you have Python running in some kind of virtual environment (e.g. 
[uv](https://docs.astral.sh/uv/guides/install-python/), [conda](https://docs.conda.io/projects/conda/en/stable/), [miniconda](https://docs.conda.io/en/latest/miniconda.html),
[venv](https://realpython.com/python-virtual-environments-a-primer/))
2. Install the `pymoto` Python package (and its dependencies)
   - Option A (pip): Type `pip install pymoto` into your console to install (prepend with `uv` when using `uv`)
   - Option B (conda): If you are working with Conda, install by `conda install -c aatmdelissen pymoto`
3. Optional: Install Intel MKL library for a fast linear solver with `pip install mkl`
   - This can also be achieved automatically by installing `pymoto` as `pip install pymoto[fast]`

After installation, `pymoto` can be used.
- Examples can be found and downloaded from the [pyMOTO examples gallery](https://pymoto.readthedocs.io/en/latest/auto_examples/index.html)
- Run the example by typing `python ex_name_of_the_example.py` in the console (prepend with `uv run` when using `uv`)

# Citing pyMOTO
When your research uses `pymoto`, please consider citing out Zenodo entry in any publications: 
[DOI:10.5281/zenodo.8138859](https://doi.org/10.5281/zenodo.8138859).

## Dependencies
* [**numpy**](https://numpy.org/doc/stable/) - Dense linear algebra and solvers
* [**scipy**](https://docs.scipy.org/doc/scipy/) - Sparse linear algebra and solvers
* [**sympy**](https://docs.sympy.org/latest/index.html) - Symbolic differentiation for `MathExpression` module
* [**Matplotlib**](https://matplotlib.org/stable/) - Plotting and visualisation
* (optional) [**opt_einsum**](https://optimized-einsum.readthedocs.io/en/stable/install.html) - Optimized function for `EinSum` module

For fast linear solvers for sparse matrices:
* (optional) [**mkl**](https://pypi.org/project/mkl) - Use the Intel OneAPI PARDISO solver (recommended)
* (optional) [**scikit-umfpack**](https://scikit-umfpack.github.io/scikit-umfpack/install.html) - Fast LU linear solver based on UMFPACK
* (optional) [**scikit-sparse**](https://github.com/scikit-sparse/scikit-sparse) - Fast Cholesky solver based on CHOLMOD
* (optional) [**cvxopt**](https://cvxopt.org/install/index.html) - Another fast Cholesky solver based on CHOLMOD

__Note on linear solvers for sparse matrices:__ Scipy implements a version of LU which is quite slow. To increase the  speed of the optimization, `mkl` is recommended as it contains PARDISO, which is a very robust and flexible solver for  any matrix (symmetric, asymmetric, real, or complex). An alternative is `scikit-umfpack` which provides a fast LU factorization. For symmetric matrices a Cholesky factorization can be used (not provided with Scipy), by either installing `scikit-sparse` or `cvxopt`.

# Contributing
For development, a local installation of `pymoto` can be done by first downloading/cloning the entire git repo, and then calling `pip install -e .` in the `pymoto` folder (of course from within your virtual environment). This allows making changes to the `pymoto` code without having to reinstall.

You are now ready for a contribution to `pymoto`.
1. Check the [issues page](https://github.com/aatmdelissen/pyMOTO/issues) to see if the subject you want to improve is listed:
   - If an issue already exists, add your view to the problem as a comment and let us know you are working on this.
   - Open a new issue if it is not listed, and discuss your ideas.
2. Create a new branch to work the code, the branch can also be in your own fork of the repo.
3. Work that code.
4. Make sure all tests are passed (by running `pytest`) and style is adhered
5. Create a [pull request](https://github.com/aatmdelissen/pyMOTO/pulls) describing your changes

Thanks for your contribution! We will have it reviewed for it to be merged with the main code.

# License
pyMOTO is available under te [MIT License](https://opensource.org/licenses/MIT).
