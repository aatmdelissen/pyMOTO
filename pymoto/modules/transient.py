import numpy as np
from pymoto import Module, DyadCarrier
from pymoto.solvers import auto_determine_solver, LDAWrapper

class TransientThermal(Module):
    r""" Solves the transient thermal problem :math:`\mathbf{K}\mathbf{T} + \mathbf{C}\dot{\mathbf{T}} = \mathbf{Q}`

    Solves the transient thermal problem :math:`\mathbf{K}\mathbf{T} + \mathbf{C}\dot{\mathbf{T}} = \mathbf{Q}`, which
    becomes :math:`(1-\theta)\mathbf{Q}^{\text{n-1}} + \theta\mathbf{Q}^{\text{n}} =
    (-\frac{\mathbf{C}}{\Delta t} + (1-\theta)\mathbf{K})\mathbf{T}^{\text{n-1}} +
    (\frac{\mathbf{C}}{\Delta t} + \theta\mathbf{K})\mathbf{T}^{\text{n}}` for timestep ``n`` using numerical
    timestepping.

    Input Signals:
      - ``K`` (`dense or sparse matrix`): The system matrix :math:`\mathbf{K}` of size ``(n, n)``
      - ``C`` (`dense or sparse matrix`): The damping matrix :math:`\mathbf{C}` of size ``(n, n)``
      - ``Q`` (`dense matrix`): Right-hand-side matrix of size ``(n, Ntimesteps)``

    Output Signals:
      - ``T`` (`matrix`): Solution matrix of size ``(n, Ntimesteps)``

    Keyword Args:
        T_0: Initial temperature vector of size (n)
        end: End time of transient simulation
        dt: Size of time step
        theta: Time-stepping algorithm, 0.0 for forward Euler, 0.5 for Crank-Nicolson, 1.0 for backward Euler
        solver: Manually override the LinearSolver used, instead of the solver from :func:`auto_determine_solver`
        """
    def _prepare(self, T_0, end, dt, theta = 1.0, solver = None):
        self.T_0 = T_0
        self.steps = int(end/dt)
        self.dt = dt
        self.theta = theta
        assert 0.0 <= self.theta <= 1.0, "theta must be between 0.0 and 1.0"
        self.solver = solver

    def _response(self, K, C, Q):
        # prepare matrices for solve
        C_step = C.multiply(1 / self.dt)
        K_forward = K.multiply(self.theta)
        K_backward = K.multiply(1 - self.theta)
        self.mat_forward = K_forward + C_step
        self.mat_backward = K_backward - C_step

        # Determine the solver we want to use
        if self.solver is None:
            self.solver = auto_determine_solver(self.mat_forward)
        if not isinstance(self.solver, LDAWrapper):
            lda_kwargs = dict(hermitian=True, symmetric=True)
            if hasattr(self.solver, 'tol'):
                lda_kwargs['tol'] = self.solver.tol * 5
            self.solver = LDAWrapper(self.solver, **lda_kwargs)

        # Update solver with new matrix
        self.solver.update(self.mat_forward)

        # initialize temperatures
        self.temperature = np.zeros((self.mat_forward.shape[0], self.steps + 1))
        self.temperature[:, 0] = self.T_0

        # perform solve for every time step
        for i in range(self.steps):
            rhs = self.theta * Q[:, i + 1] + (1 - self.theta) * Q[:, i] - self.mat_backward.dot(self.temperature[:, i])
            self.temperature[:, i + 1] = self.solver.solve(rhs)

        return self.temperature

    def _sensitivity(self, dfdt):
        # initialize adjoint variables
        lams = np.zeros_like(self.temperature)
        lams[:, -1] = self.solver.solve(-dfdt[:, -1])

        # perform adjoint solve for every time step
        for i in reversed(range(self.steps)):
            rhs = -dfdt[:, i] - self.mat_backward.dot(lams[:, i + 1])
            lams[:, i] = self.solver.solve(rhs)

        # sensitivities to system matrices
        tempsK = self.theta*self.temperature[:, 1:] + (1-self.theta)*self.temperature[:, :-1]
        dK = DyadCarrier(list(lams[:, 1:].T), list(tempsK.T))
        dC = DyadCarrier(list(lams[:, 1:].T), list((1/self.dt)*np.diff(self.temperature).T))

        # sensitivities to input heat
        dQ = np.zeros_like(self.sig_in[2].state)
        dQ[:, 1:] -= self.theta*lams[:, 1:]
        dQ[:, :-1] -= (1-self.theta)*lams[:, 1:]

        return dK, dC, dQ