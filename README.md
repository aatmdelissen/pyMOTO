![logo](M_logo_256.png)

Modular framework for optimization with semi-automatic derivatives

# Requirements

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

# Installation
Download the git repository and unzip
cd to the downloaded directory

Run the following command to install the package:
```bash
python setup.py install
```

After installing, execute tests by typing:
```bash
python setup.py test
```

# License
PyModular is available under te [MIT License](https://opensource.org/licenses/MIT).
