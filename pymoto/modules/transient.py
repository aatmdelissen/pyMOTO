import numpy as np
from pymoto import Module, DyadCarrier, Signal, LinSolve

class TransientThermal(Module):
    def _prepare(self, T_0, end, dt, theta = 1.0, **kwargs):
        self.T_0 = T_0
        self.steps = end/dt
        self.dt = dt
        self.theta = theta
        self.module_LinSolve = LinSolve([Signal(), Signal()], **kwargs)