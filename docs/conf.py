# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Run `uv run sphinx-build . _build` from within the docs/ directory to build the documentation
import os
import sys
import datetime
import shutil
sys.path.insert(0, os.path.abspath('..'))
import pymoto

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyMOTO"
author = "Arnoud Delissen"
copyright = f"{datetime.date.today().year}, {author}"
release = pymoto.__version__
version = release[:release.find('.', 2)]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # To parse markdown files
    # "myst_nb",
    "sphinx.ext.autodoc",  # To automatically generate documentation from docstrings
    "sphinx.ext.napoleon",  # To support Google and NumPy style docstrings
    "sphinxcontrib.mermaid",  # For flowcharts and diagrams
    "sphinx.ext.autosummary",  # Generates function/method/attribute summary lists
    # "myst_sphinx_gallery",  # For example gallery
    "sphinx_gallery.gen_gallery", # For generating the example gallery
]

# https://thebe.readthedocs.io/en/latest/# for interactive jupyter
source_suffix = {
    '.rst': 'restructuredtext',
    # '.txt': 'markdown',
    '.md': 'markdown',
    '.tex': 'markdown',
    # '.md': 'myst-nb',
    # '.tex': 'myst-nb',
    # '.ipynb': 'myst-nb',
}
templates_path = ['_templates']
exclude_patterns = ['developer']

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'line_numbers': True,
     'nested_sections': False,
}

# -- Copy examples into _examples --------------------------------------------
# examples_folder = '../examples'
# target_folder = './_collections/examples'
# shutil.copytree(examples_folder, target_folder, dirs_exist_ok=True)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['sphinx_gallery_custom.css']
html_logo = "_static/M_logo.svg"
html_favicon = "_static/M_logo_16.png"

# Napoleon settings
napoleon_numpy_docstring = False
napoleon_google_docstring = True

# Myst settings for markdown support
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
