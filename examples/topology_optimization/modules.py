import numpy as np

import pymoto as pym
from pymoto.modules.assembly import get_B, get_D


# region Stress
class Stress(pym.Module):
    def _prepare(self, E=1, nu=0.3, plane='strain', domain=pym.DomainDefinition, *args, **kwargs):
        siz = domain.element_size
        self.domain = domain

        # Constitutive model
        self.D = siz[2] * get_D(E, nu, plane.lower())

        # Numerical integration
        self.B = np.zeros((3, 8), dtype=float)
        w = np.prod(siz[:domain.dim] / 2)
        for n in domain.node_numbering:
            pos = n * (siz / 2) / np.sqrt(3)  # Sampling point
            dN_dx = domain.eval_shape_fun_der(pos)
            self.B += w * get_B(dN_dx)

        self.dofconn = domain.get_dofconnectivity(2)

    def _response(self, u):
        self.elemental_strain = self.B.dot(u[self.dofconn].transpose())
        self.elemental_strain[2, :] *= 2  # voigt notation
        return self.D.dot(self.elemental_strain).transpose()

    def _sensitivity(self, dfdv):
        dgdsstrainmat = np.einsum('jk,kl->jl', dfdv, self.D)
        dgdsstrainmat[:, 2] *= 2
        dgdue = np.einsum('ij,jl->il', dgdsstrainmat, self.B)

        y = np.zeros(self.domain.nnodes * 2)
        for i in range(0, self.domain.nel):
            y[self.dofconn[i, :]] += dgdue[i, :]
        return y


class VonMises(pym.Module):
    def _prepare(self, *args, **kwargs):
        # Vandermonde matrix
        self.V = np.array([[1, -0.5, 0], [-0.5, 1, 0], [0, 0, 3]])

    def _response(self, x):
        self.x = x
        self.y = (x.dot(self.V) * x).sum(1)
        return np.sqrt(self.y)

    def _sensitivity(self, dfdv):
        return dfdv[:, np.newaxis] * (self.y ** (-0.5))[:, np.newaxis] * self.x.dot(self.V)


class ConstraintAggregation(pym.Module):
    """
    Unified aggregation and relaxation.

    Implemented by @artofscience (s.koppen@tudelft.nl), based on:

    Verbart, A., Langelaar, M., & Keulen, F. V. (2017).
    A unified aggregation and relaxation approach for stress-constrained topology optimization.
    Structural and Multidisciplinary Optimization, 55, 663-679.
    DOI: https://doi.org/10.1007/s00158-016-1524-0
    """

    def _prepare(self, P=10):
        self.P = P

    def _response(self, x):
        """
        a = x + 1
        b = aggregation(a)
        c = b - 1
        """
        self.n = len(x)
        self.x = x
        self.y = self.x + 1
        self.z = self.y ** self.P
        z = ((1 / len(self.x)) * np.sum(self.z)) ** (1 / self.P)  # P-mean aggregation function
        return z - 1

    def _sensitivity(self, dfdc):
        return (dfdc / self.n) * (np.sum(self.z) / self.n) ** (1 / self.P - 1) * self.y ** (self.P - 1)


# endregion

# region Dynamics
class DynamicMatrix(pym.Module):
    """ Constructs dynamic stiffness matrix with Rayleigh damping """

    def _prepare(self, alpha=0.5, beta=0.5):
        self.alpha = alpha
        self.beta = beta

    def _response(self, K, M, omega):
        return K + 1j * omega * (self.alpha * M + self.beta * K) - omega ** 2 * M

    def _sensitivity(self, dZ: pym.DyadCarrier):
        K, M, omega = [s.state for s in self.sig_in]
        dZr, dZi = dZ.real, dZ.imag
        dK = dZr - (omega * self.beta) * dZi
        dM = (-omega ** 2) * dZr - (omega * self.alpha) * dZi
        dZrM = dZr.contract(M)
        dZiK = dZi.contract(K)
        dZiM = dZi.contract(M)
        domega = -self.beta * dZiK - self.alpha * dZiM - 2 * omega * dZrM
        return dK, dM, domega


# endregion

class SelfWeight(pym.Module):
    def _prepare(self, gravity=np.array([0.0, -1.0], dtype=float), domain=pym.DomainDefinition):
        self.load_x = gravity[0] / 4
        self.load_y = gravity[1] / 4
        self.dofconn = domain.get_dofconnectivity(2)
        self.f = np.zeros(domain.nnodes * 2, dtype=float)
        self.dfdx = np.zeros(domain.nel, dtype=float)

    def _response(self, x, *args):
        self.f[:] = 0.0
        load_x = np.kron(x, self.load_x * np.ones(4))
        load_y = np.kron(x, self.load_y * np.ones(4))
        np.add.at(self.f, self.dofconn[:, 0::2].flatten(), load_x)
        np.add.at(self.f, self.dofconn[:, 1::2].flatten(), load_y)
        return self.f

    def _sensitivity(self, dfdv):
        self.dfdx[:] = 0.0
        self.dfdx[:] += dfdv[self.dofconn[:, 0::2]].sum(1) * self.load_x
        self.dfdx[:] += dfdv[self.dofconn[:, 1::2]].sum(1) * self.load_y
        return self.dfdx


class Continuation(pym.Module):
    """ Module that generates a continuated value """

    def _prepare(self, start=0.0, stop=1.0, nsteps=80, stepstart=10):
        self.startval = start
        self.endval = stop
        self.dval = (stop - start) / nsteps
        self.nstart = stepstart
        self.iter = -1
        self.val = self.startval

    def _response(self):
        if (self.val < self.endval and self.iter > self.nstart):
            self.val += self.dval

        self.val = np.clip(self.val, min(self.startval, self.endval), max(self.startval, self.endval))
        print(self.sig_out[0].tag, ' = ', self.val)
        self.iter += 1
        return self.val

    def _sensitivity(self, *args):
        pass


class Symmetry(pym.Module):
    def _prepare(self, domain=pym.DomainDefinition):
        self.domain = domain

    def _response(self, x):
        x = np.reshape(x, (self.domain.nely, self.domain.nelx))
        x = (x + np.flip(x, 0)) / 2
        x = (x + np.flip(x, 1)) / 2
        return x.flatten()

    def _sensitivity(self, dfdv):
        dfdv = np.reshape(dfdv, (self.domain.nely, self.domain.nelx))
        dfdv = (dfdv + np.flip(dfdv, 1)) / 2
        dfdv = (dfdv + np.flip(dfdv, 0)) / 2
        return dfdv


class VecSet(pym.Module):
    def _prepare(self, indices, value):
        self.indices = indices
        self.value = value

    def _response(self, x):
        y = x.copy()
        y[self.indices] = self.value
        return y

    def _sensitivity(self, dy):
        dx = dy.copy()
        dx[self.indices] = 0
        return dx
