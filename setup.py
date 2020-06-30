from setuptools import setup

setup(name='pyModular',
      version='0.1',
      description='Framework for optimization',
      long_description='Semi-automatic differentiation',
      keywords='optimization framework modular blocks',
      url='https://github.com/aatmdelissen/pyModular',
      author='Arnoud Delissen',
      author_email='a.a.t.m.delissen@tudelft.nl',
      license='',
      packages=['pyModular'],
      install_requires=['numpy', 'sympy'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose']
      )
