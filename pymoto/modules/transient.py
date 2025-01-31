import numpy as np
from pymoto import Module, DyadCarrier, Signal, LinSolve

class TransientThermal(Module):
    def _prepare(self, T_0, end, dt, theta = 1.0, **kwargs):
        self.T_0 = T_0
        self.steps = end/dt
        self.dt = dt
        self.theta = theta
        self.module_LinSolve = LinSolve([Signal(), Signal()], **kwargs)

    def _response(self, K, C, Q):
        # prepare matrices for solve
        Cstep = C.multiply(1 / self.dt)
        K_forward = K.multiply(self.theta)
        K_backward = K.multiply(1 - self.theta)
        self.mat_forward = K_forward + Cstep
        self.mat_backward = K_backward - Cstep
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