from setuptools import setup

setup(name='pymodular',
      version='0.1',
      description='Framework for optimization',
      long_description='Semi-automatic differentiation',
      keywords='optimization framework modular blocks',
      url='https://github.com/aatmdelissen/pyModular',
      author='Arnoud Delissen',
      author_email='a.a.t.m.delissen@tudelft.nl',
      license='',
      packages=['pymodular'],
      install_requires=['numpy', 'sympy', 'scipy', 'matplotlib'],
      zip_safe=False,
      )
