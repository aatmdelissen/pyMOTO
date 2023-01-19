from setuptools import setup

setup(name='pymoto',
      version='1.0',
      description='A modular framework to perform Topology Optimization in a flexible manner',
      long_description='Semi-automatic differentiation',
      keywords='topology optimization framework modular blocks',
      url='https://github.com/aatmdelissen/pyModular',
      author='Arnoud Delissen',
      author_email='arnouddelissen@gmail.com',
      license='MIT',
      packages=['pymoto'],
      install_requires=['numpy', 'sympy', 'scipy', 'matplotlib'],
      zip_safe=False,
      )
