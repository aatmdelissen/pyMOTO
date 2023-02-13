from setuptools import setup
from pymoto import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name='pymoto',
      version=__version__,
      author='Arnoud Delissen',
      author_email='arnouddelissen+pymoto@gmail.com',
      description='A modular framework to perform Topology Optimization in a flexible manner',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='Topology Optimization Framework Modular Blocks Pipeline Structural Generative Design',
      url='https://github.com/aatmdelissen/pyMOTO',
      packages=['pymoto'],
      package_data={'pymoto': ['*', 'common/*', 'modules/*']},
      install_requires=['numpy', 'sympy', 'scipy>=1.7', 'matplotlib'],
      classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering"
      ],
)
