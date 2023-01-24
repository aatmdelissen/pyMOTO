"""
Definition of adjoint variable for :math:`z = x+iy` :

  - :math:`\\frac{\partial}{\partial z} = \\frac{1}{2}\left( \\frac{\partial}{\partial x} - i \\frac{\partial}{\partial y}\\right)` (<-- this is stored in the adjoint variable (:attr:`Signal.sensitivity`) to keep the the rule ``adj_z`` := ∂f/∂z )
  - :math:`\\frac{\partial}{\partial z^*} = \\frac{1}{2}\left( \\frac{\partial}{\partial x} + i \\frac{\partial}{\partial y}\\right)`

Useful identities (:math:`z\in\mathbb{C}` and :math:`s\in\mathbb{C}`)

  - :math:`\\frac{\partial s^*}{\partial z^*} = \left( \\frac{\partial s}{\partial z} \\right)^*`
  - :math:`\\frac{\partial s^*}{\partial z} = \left( \\frac{\partial s}{\partial z^*} \\right)^*`

In case :math:`f\in\mathbb{R}`

  - :math:`\\frac{\partial f}{\partial z^*} = \left( \\frac{\partial f}{\partial z} \\right)^*`

The chain rule for a mapping from :math:`z\in\mathbb{C}\\rightarrow s\in\mathbb{C}\\rightarrow f` can be seen as the
contributions of two independent variables, e.g. :math:`s` and :math:`s^*`:

:math:`\\frac{\partial f}{\partial z} = \\frac{\partial f}{\partial s}\\frac{\partial s}{\partial z} + \\frac{\partial f}{\partial s^*}\\frac{\partial s^*}{\partial z}`

In case the intermediate variable is real, thus :math:`z\in\mathbb{C}\\rightarrow r\in\mathbb{R}\\rightarrow f`, the
chain rule reduces to

:math:`\\frac{\partial f}{\partial z} = 2 \\frac{\partial f}{\partial r}\\frac{\partial r}{\partial z}`,

which may seem counter-intuitive, but compensates for the initial factor of 1/2.

For a mapping from real to complex, thus :math:`r\in\mathbb{R}\\rightarrow z\in\mathbb{C}\\rightarrow f`, the chain rule
becomes

:math:`\\frac{\partial f}{\partial z} = 2 \\text{Re}\left( \\frac{\partial f}{\partial s}\\frac{\partial s}{\partial r} \\right)`.

References:
  - Sarason (2007). Complex function theory. American Mathematical Society.
  - `Delgado (2009). The complex gradient operator and the CR-calculus <https://arxiv.org/pdf/0906.4835.pdf>`_
  - `Pytorch AutoGrad <https://pytorch.org/docs/stable/notes/autograd.html>`_
"""

import numpy as np
from pymoto import Module


class MakeComplex(Module):
    """ Makes a complex variable from two real inputs :math:`z(x,y) = x + iy` """
    def _response(self, x, y):
        return x + 1j*y

    def _sensitivity(self, dz):
        return np.real(dz), np.real(1j*dz)


class RealPart(Module):
    """ Takes the real part of a complex value :math:`x = \\text{Re}(z)` """
    def _response(self, z):
        return np.real(z)

    def _sensitivity(self, dx):
        return np.real(dx)


class ImagPart(Module):
    """ Takes the imaginary part of a complex value :math:`y = \\text{Im}(z)` """
    def _response(self, z):
        return np.imag(z)

    def _sensitivity(self, dy):
        return -1j*dy


class ComplexNorm(Module):
    """ Takes the complex norm :math:`n = z z^*` """
    def _response(self, z):
        return np.absolute(z)

    def _sensitivity(self, dA):
        z = self.sig_in[0].state
        A = self.sig_out[0].state
        return 1/A*dA*np.conj(z)
