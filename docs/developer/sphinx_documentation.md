# Sphinx documentation

## Setup environment
`pip install sphinx` for documentation

`pip install sphinx_rtd_theme` for nice looking website

`pip install m2r` or `pip install myst-parser` for markdosn (and latex formula) support

`pip install sphinxcontrib-mermaid` for graphs

`mkdir docs` Initial setup

`sphinx-quickstart` Initial automatic setup for directory and file structure

## Make documentation
`cd docs/` Navigate to the documentation folder

`sphinx-apidoc -f -o source/ ../pymoto/`

`make html`