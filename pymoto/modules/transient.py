import numpy as np
from pymoto import Module, DyadicMatrix
from pymoto.solvers import auto_determine_solver, LDAWrapper


class TransientSolve(Module):
    r"""Solves the transient thermal problem :math:`\mathbf{K}\mathbf{T} + \mathbf{C}\dot{\mathbf{T}} = \mathbf{Q}`

    Solves the transient thermal problem :math:`\mathbf{K}\mathbf{T} + \mathbf{C}\dot{\mathbf{T}} = \mathbf{Q}`, which
    becomes :math:`(1-\theta)\mathbf{Q}^{\text{n-1}} + \theta\mathbf{Q}^{\text{n}} =
    (-\frac{\mathbf{C}}{\Delta t} + (1-\theta)\mathbf{K})\mathbf{T}^{\text{n-1}} +
    (\frac{\mathbf{C}}{\Delta t} + \theta\mathbf{K})\mathbf{T}^{\text{n}}` for timestep ``n`` using numerical
    timestepping.

    Input Signals:
      - ``b`` (`dense matrix or vector`): Right-hand-side matrix of size ``(n, Ntimesteps)`` or vector of size ``(n)``
      - ``K`` (`dense or sparse matrix`): The system matrix :math:`\mathbf{K}` of size ``(n, n)``
      - ``C`` (`dense or sparse matrix`): The damping matrix :math:`\mathbf{C}` of size ``(n, n)``

    Output Signals:
      - ``state`` (`matrix`): Solution matrix of size ``(n, Ntimesteps)``
    """

    def __init__(self, dt, end=None, x0=None, theta=1.0, solver=None):
        """Initialize the transient solver module

        Args:
          - `dt` (float, required): Size of time step (in seconds)
          - `end`(float, optional): End time of simulation (in seconds). If not provided, the number of time steps is
            deduced from the size of the input matrix ``b``
          - `x0` (vector, optional): Initial state of size ``(n)``
          - `theta` (float, optional): Time-stepping algorithm, `0.0` for forward Euler, `0.5` for Crank-Nicolson, `1.0`
            for backward Euler
          - `solver` (:py:class:`pymoto.solvers.LinearSolver`, optional): Manually override the linear solver used,
            instead of the solver from :func:`pymoto.solvers.auto_determine_solver`
        """
        self.x0 = x0
        self.end = end
        self.dt = dt
        self.theta = theta
        assert 0.0 <= self.theta <= 1.0, "theta must be between 0.0 and 1.0"
        self.solver = solver

    def __call__(self, b, K, C, M=None):
        # initial checks
        if M is not None:
            raise NotImplementedError("Mass matrix not implemented yet")
        assert K.shape == C.shape, "K and C must have same shape"
        if self.end is None and b.ndim != 1:
            raise RuntimeError("Insufficient information to determine time steps. Provide `end` time or a matrix `b`")

        # determine time steps
        if b.ndim == 1:
            self.steps = int(self.end / self.dt)
            b = np.tile(b, (self.steps + 1, 1)).T
        else:
            self.steps = b.shape[1] - 1

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
            if hasattr(self.solver, "tol"):
                lda_kwargs["tol"] = self.solver.tol * 5
            self.solver = LDAWrapper(self.solver, **lda_kwargs)

        # Update solver with new matrix
        self.solver.update(self.mat_forward)

        # initialize temperatures
        self.state = np.zeros((self.mat_forward.shape[0], self.steps + 1))
        if self.x0 is not None:
            self.state[:, 0] = self.x0

        # perform solve for every time step
        for i in range(self.steps):
            rhs = self.theta * b[:, i + 1] + (1 - self.theta) * b[:, i] - self.mat_backward.dot(self.state[:, i])
            self.state[:, i + 1] = self.solver.solve(rhs)

        return self.state

    def _sensitivity(self, dfdt):
        # initialize adjoint variables
        lams = np.zeros_like(self.state)
        lams[:, -1] = self.solver.solve(-dfdt[:, -1])

        # perform adjoint solve for every time step
        for i in reversed(range(self.steps)):
            rhs = -dfdt[:, i] - self.mat_backward.dot(lams[:, i + 1])
            lams[:, i] = self.solver.solve(rhs)

        # sensitivities to system matrices
        tempsK = self.theta * self.state[:, 1:] + (1 - self.theta) * self.state[:, :-1]
        dK = DyadicMatrix(list(lams[:, 1:].T), list(tempsK.T))
        dC = DyadicMatrix(list(lams[:, 1:].T), list((1 / self.dt) * np.diff(self.state).T))

        # sensitivities to input heat
        db = np.zeros_like(self.sig_in[0].state)
        db[:, 1:] -= self.theta * lams[:, 1:]
        db[:, :-1] -= (1 - self.theta) * lams[:, 1:]

        return db, dK, dC
