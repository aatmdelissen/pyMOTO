# Sphinx documentation
Guide for developers to generate documentation for pyMOTO.

## Setup environment
All requirements for generating documentation are listed in `requirements.txt`. To install these packages just run

```pip install -r requirements.txt```

### Package explanation
`pip install sphinx` for documentation

`pip install sphinx_rtd_theme` for nice looking website

`pip install m2r` or `pip install myst-parser` for markdown (and latex formula) support

`pip install sphinxcontrib-mermaid` for graphs

## Initial setup of documentation
`mkdir docs` Initial setup

`sphinx-quickstart` Initial automatic setup for directory and file structure

## Make documentation
`cd docs/` Navigate to the documentation folder

`sphinx-apidoc -f -o source/ ../pymoto/`

`make html`