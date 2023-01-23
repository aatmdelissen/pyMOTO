# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import pymoto

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyMOTO'
copyright = '2023, Arnoud Delissen'
author = 'Arnoud Delissen'
release = pymoto.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['m2r', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon']
source_suffix = ['.rst', '.md']
templates_path = ['_templates']
exclude_patterns = []

napoleon_numpy_docstring = False
napoleon_google_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']