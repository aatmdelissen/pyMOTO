import numpy as np
from pymoto import Module, DyadCarrier, Signal, LinSolve

class TransientThermal(Module):
    def _prepare(self, T_0, end, dt, theta = 1.0, **kwargs):
        self.T_0 = T_0
        self.steps = end/dt
        self.dt = dt
        self.theta = theta
        assert 0.0 <= self.theta <= 1.0, "theta must be between 0.0 and 1.0"
        self.module_LinSolve = LinSolve([Signal(), Signal()], **kwargs)

    def _response(self, K, C, Q):
        # prepare matrices for solve
        C_step = C.multiply(1 / self.dt)
        K_forward = K.multiply(self.theta)
        K_backward = K.multiply(1 - self.theta)
        self.mat_forward = K_forward + C_step
        self.mat_backward = K_backward - C_step
        self.module_LinSolve.sig_in[0].state = self.mat_forward

        # initialize temperatures
        self.temperature = np.zeros((self.mat_forward.shape[0], self.steps + 1))
        self.temperature[:, 0] = self.T_0

        # perform solve for every time step
        for i in range(self.steps):
            rhs = self.theta * Q[:, i + 1] + (1 - self.theta) * Q[:, i] - self.mat_backward.dot(self.temperature[:, i])
            self.module_LinSolve.sig_in[1].state = rhs
            self.module_LinSolve.response()
            self.temperature[:, i + 1] = self.module_LinSolve.sig_out[0].state

        return self.temperature

    def _sensitivity(self, dfdt):
        # initialize adjoint variables
        lams = np.zeros_like(self.temperature)
        self.module_LinSolve.sig_in[1].state = -dfdt[:, -1]
        self.module_LinSolve.response()
        lams[:, -1] = self.module_LinSolve.sig_out[0].state

        # perform adjoint solve for every time step
        for i in reversed(range(self.steps)):
            rhs = -dfdt[:, i] - self.mat_backward.dot(lams[:, i+1])
            self.module_LinSolve.sig_in[1].state = rhs
            self.module_LinSolve.response()
            lams[:, i] = self.module_LinSolve.sig_out[0].state

        # sensitivities to system matrices
        tempsK = self.theta*self.temperature
        tempsK[:, 1:] += (1-self.theta)*self.temperature[:, :-1]
        dK = DyadCarrier(list(lams.T), list(tempsK.T))
        dC = DyadCarrier(list(lams[:, 1:].T), list((1/self.dt)*np.diff(self.temperature).T))

        # sensitivities to input heat
        dQ = np.zeros_like(self.sig_in[2])
        dQ[:, 1:] -= self.theta*lams[:, 1:]
        dQ[:, :-1] += -(1-self.theta)*lams[:, 1:]

        return dK, dC, dQ