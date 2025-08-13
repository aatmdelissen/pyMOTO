# Sphinx documentation
Guide for developers to generate documentation for pyMOTO.

## Setup environment
All requirements for generating documentation are listed in `requirements.txt`. To install these packages just run

```pip install -r requirements.txt```

## Initial setup of documentation (only needs to be done once for a new projects)
`mkdir docs` Initial setup

`sphinx-quickstart` Initial automatic setup for directory and file structure

## Make documentation
`cd docs/` Navigate to the documentation folder

`make html` or `sphinx-build . _build` (prepend with `uv run` when using uv)
Additional options: [sphinx-build](https://www.sphinx-doc.org/en/master/man/sphinx-build.html)