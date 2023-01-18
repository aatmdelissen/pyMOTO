"""
Definition of adjoint variable for z = x+iy :
∂/∂z  = 1/2(∂/∂x - i ⋅ ∂/∂y) (<-- this is stored in the adjoint variable to keep the the rule adj_z := ∂f/∂z )
∂/∂z^* = 1/2(∂/∂x + i ⋅ ∂/∂y) 

Useful identities (z∈C and s∈C)
∂s^* / ∂z^* = (∂s/∂z)^*
∂s^* / ∂z   = (∂s/∂z^*)^*
In case f∈R
∂f/∂z^* = (∂f/∂z)^*

Chain rule: z∈C -> s∈C: ∂f/∂z = ∂f/∂s ⋅ ∂s/∂z + ∂f/∂s^*     ⋅ ∂s^*/∂z 
                        z_adj = s_adj ⋅ ∂s/∂z + conj(s_adj) ⋅ d∂^*/∂z
                        
Mapping for z∈C -> r∈R: ∂f/∂z = 2 ⋅ ∂f/dr ⋅ ∂r/∂z
                        z_adj = 2 ⋅ s_adj ⋅ ∂s/∂z
                        
Mapping for r∈R -> z∈C: ∂f/∂r = 2 ⋅ Re( ∂f/∂s ⋅ ∂s/∂r )
                        z_adj = 2 ⋅ Re( s_adj ⋅ ∂s/∂r )

References: 
https://pytorch.org/docs/stable/notes/autograd.html
https://arxiv.org/pdf/0906.4835.pdf
"""

import numpy as np
from pymodular import Module


class MakeComplex(Module):
    """ Makes a complex variable from two real inputs z = x + jy """
    def _response(self, x, y):
        return x + 1j*y

    def _sensitivity(self, dz):
        return np.real(dz), np.real(1j*dz)


class RealPart(Module):
    """ Takes the real part of a complex value x = Re(z) """
    def _response(self, z):
        return np.real(z)

    def _sensitivity(self, dx):
        return np.real(dx)


class ImagPart(Module):
    """ Takes the imaginary part of a complex value y = Im(z) """
    def _response(self, z):
        return np.imag(z)

    def _sensitivity(self, dy):
        return -1j*dy


class ComplexNorm(Module):
    """ Takes the complex norm """
    def _response(self, z):
        return np.absolute(z)

    def _sensitivity(self, dA):
        z = self.sig_in[0].state
        A = self.sig_out[0].state
        return 1/A*dA*np.conj(z)
