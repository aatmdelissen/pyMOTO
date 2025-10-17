import numpy as np
from .optimizers import Optimizer


class MMA(Optimizer):
    r"""Class for the Method of Moving Asymptotes (MMA) optimization algorithm
    
    References:
      - Svanberg, K. (1987). The Method of Moving Asymptotes. IJNME, 24(2), 359–373. 
        https://doi.org/10.1002/nme.1620240207
      - Svanberg, K. (2007). MMA and GCMMA – two methods for nonlinear optimization. Kth, 1, 1–15.
    """

    def __init__(
        self,
        variables,
        responses,
        function,
        slice_network=False,
        move=0.1,
        xmin=0.0,
        xmax=1.0,
        verbosity=2,
        mmaversion="MMA2007",
        **kwargs,
    ):
        """
        Args:
          variables: One or more variable Signals defining the design variables
          responses: One or more response Signals, where the first is to be minimized and the others are constraints 
            in negative null form.
          function: The Network defining the optimization problem

        Keyword Args:
          slice_network: If True, only the modules connecting variable and response signals are evaluated
          move: Move limit on relative variable change per iteration (can be passed as scalar: same value for all 
            variables, vector: each variable has a unique value, list of scalars/vectors: for each variable signal)
          xmin: Minimum design variable (can be passed as scalar: same value for all variables, vector: each variable 
            has a unique value, list of scalars/vectors: for each variable signal)
          xmax: Maximum design variable (can be passed as scalar: same value for all variables, vector: each variable 
            has a unique value, list of scalars/vectors: for each variable signal)
          verbosity: Level of information to print
            0 - No prints
            1 - Only convergence message
            2 - Convergence and iteration info (default)
            3 - Additional info on variables and GCMMA inner iteration info (when applicable)
            4 - Additional info on sensitivity information
          mmaversion: Which MMA algorithm to use
            "MMA1987" - Original version of MMA
            "MMA2007" - Improved version of MMA
            "GCMMA" - Globally-convergent version of MMA
          **kwargs: Additional MMA options (see code for details)
        """
        super().__init__(variables, responses, function, slice_network=slice_network, 
                         xmin=xmin, xmax=xmax, verbosity=verbosity)

        # Operational options
        if np.asarray(move).size > 1:
            self.move = self._parse_bound(move, which='move')
        else:
            self.move = move

        # MMA-specific options
        self.mmaversion = mmaversion  # Options are MMA1987, MMA2007, GCMMA

        self.a0 = kwargs.get("a0", 1.0)
        self.epsimin = kwargs.get("epsimin", 1e-10)  # Or 1e-7 ?? witout sqrt(m+n) or 1e-9
        self.cCoef = kwargs.get("cCoef", 1e3)  # Svanberg uses 1e3 in example? Old code had 1e7

        # Asymptotes control (MMA)
        self.albefa = kwargs.get("albefa", 0.1)
        self.asyinit = kwargs.get("asyinit", 0.5)
        self.asyincr = kwargs.get("asyincr", 1.2)
        self.asydecr = kwargs.get("asydecr", 0.7)
        self.asybound = kwargs.get("asybound", 10.0)
        
        # GCMMA options
        self.gcmma_maxit = kwargs.get("gcmma_maxit", 20)

        # Numbers
        self.dx = self.xmax - self.xmin
        self.xold1, self.xold2 = None, None
        self.low, self.upp = None, None
        self.offset = self.asyinit * np.ones(self.n)

        # Setting up for constraints
        self.m = max(1, len(self.responses) - 1) # At minimum 1 for the dummy constraint in mmasub()
        self.a = kwargs.get("a", np.zeros(self.m))
        if len(self.a) != self.m:
            raise RuntimeError(f"Length of the a vector ({len(self.a)}) should be equal to # constraints ({self.m}).")
        self.c = kwargs.get("c", np.full(self.m, self.cCoef, dtype=float))
        if len(self.c) != self.m:
            raise RuntimeError(f"Length of the c vector ({len(self.c)}) should be equal to # constraints ({self.m}).")
        self.d = np.ones(self.m)

    def step(self, x: np.ndarray = None, 
             g: np.ndarray = None, 
             dg: np.ndarray = None) -> (np.ndarray, np.ndarray, np.ndarray):
        if x is None:
            x = self.x  # Gather the states
        else:
            self.x = x  # Set the new states
        is_gcmma = "gcmma" in self.mmaversion
        max_gcmmait = self.gcmma_maxit  if is_gcmma else 1  # One iteration of GCMMA is same as MMA
        self.gest = None
        xnew = None

        # Update ASYMPTOTES
        # Calculation of the asymptotes low and upp :
        # For iter = 1,2 the asymptotes are fixed depending on asyinit
        if self.xold1 is not None and self.xold2 is not None:
            # depending on if the signs of xval - xold and xold - xold2 are opposite, indicating an oscillation
            # in the variable xi
            # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed

            # check for oscillations in variables
            # if zzz positive no oscillations, if negative --> oscillations
            zzz = (x - self.xold1) * (self.xold1 - self.xold2)
            # decrease those variables that are oscillating equal to asydecr
            self.offset[zzz > 0] *= self.asyincr
            self.offset[zzz < 0] *= self.asydecr

            # check with minimum and maximum bounds of asymptotes, as they cannot be to close or far from the variable
            # give boundaries for upper and lower asymptotes
            self.offset = np.clip(self.offset, 1 / (self.asybound**2), self.asybound)
        
        for gcmmait in range(max_gcmmait):  # Inner iterations
            if gcmmait > 0:
                self.x = xnew  # Set new design

            if g is None or gcmmait > 0:
                # Calculate and save responses
                g = self.calculate_g()

            if gcmmait == 0:
                gk = g  # Save function value of outer iteration

            if dg is None:
                # Calculate and save sensitivities (only done once for GCMMA)
                dg = self.calculate_dg()
                
            if not is_gcmma:
                self.rho = 1e-5  # No GCMMA used
            elif gcmmait > 0 and np.all(self.gest >= g):
                if self.verbosity >= 3:
                    print(f"  || GCMMA converged in {gcmmait} inner iterations")
                break
            elif gcmmait == 0:
                # Initial values
                # if hasattr(self, 'rho'):
                #     self.rho = (0.1 / self.n * np.sum(self.dx * np.abs(dg), axis=1) + self.rho)/2
                # else:
                self.rho = 0.1 / self.n * np.sum(self.dx * np.abs(dg), axis=1) 
            else:
                delta = (g - self.gest) / self.dk
                self.rho[delta > 0] = np.minimum(1.1 * (self.rho + delta), 10*self.rho)[delta > 0]
                if self.verbosity >= 3:
                    print(f"  || GCMMA It. {gcmmait}, g = {g}, gest = {self.gest}, rho={self.rho}")

            # Determine new design
            xnew = self.mmasub(x, gk, dg, rho=self.rho)
            
            # if is_gcmma:
            #     x = xnew

        self.xold2, self.xold1 = self.xold1, x.copy()

        return xnew, g, dg

    def mmasub(self, xval, g, dg, rho=1e-5):
        # Quickfix: in case of only unconstrained optimization, add a dummy constraint with zero sensitivities
        if g.size == 1:
            g = np.hstack((g, -1.0))
            dg = np.vstack((dg, np.zeros(self.n)))

        # Update asymptotes
        shift = self.offset * self.dx
        self.low = xval - shift
        self.upp = xval + shift

        # # VARIABLE BOUNDS
        # Calculation of the bounds alfa and beta :
        # use albefa to limit the maximum change of variables wrt the lower and upper asymptotes
        # as it should remain within both asymptotes
        zzl1 = self.low + self.albefa * shift
        # use movelimit to limit the maximum change of variables
        zzl2 = xval - self.move * self.dx
        # minimum variable bounds
        alfa = np.maximum.reduce([zzl1, zzl2, self.xmin])

        zzu1 = self.upp - self.albefa * shift
        zzu2 = xval + self.move * self.dx
        # maximum variable bounds
        beta = np.minimum.reduce([zzu1, zzu2, self.xmax])

        # # APPROXIMATE CONVEX SEPARABLE FUNCTIONS
        # Calculations of p0, q0, P, Q and b.
        # calculate the constant factor in calculations of pij and qij
        # From: Svanberg(2007) - MMA and GCMMA, two methods for nonlinear optimization
        dg_plus = np.maximum(+dg, 0)
        dg_min = np.maximum(-dg, 0)
        dx2 = shift**2
        if "1987" in self.mmaversion.lower():
            # Original version
            P = dx2 * dg_plus
            Q = dx2 * dg_min
        elif "2007" in self.mmaversion.lower():
            # Improved version -> Allows to use higher epsimin to get design variables closer to the bound.
            P = dx2 * (1.001 * dg_plus + 0.001 * dg_min + rho / self.dx)
            Q = dx2 * (0.001 * dg_plus + 1.001 * dg_min + rho / self.dx)
        elif "gcmma" in self.mmaversion.lower():
            # GCMMA
            P = dx2 * (1.001 * dg_plus + 0.001 * dg_min + np.maximum(rho[:, None], 1e-6) / self.dx)
            Q = dx2 * (0.001 * dg_plus + 1.001 * dg_min + np.maximum(rho[:, None], 1e-6) / self.dx)
        else:
            raise ValueError('Only "MMA1987", "MMA2007", or "GCMMA" are valid options')

        rhs = np.dot(P, 1 / shift) + np.dot(Q, 1 / shift) - g

        # Solving the subproblem by a primal-dual Newton method
        epsimin_scaled = self.epsimin * np.sqrt(self.m + self.n)
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = self.subsolv(
            epsimin_scaled, self.low, self.upp, alfa, beta, P, Q, self.a0, self.a, rhs[1:], self.c, self.d, x0=xval
        )
        
        # Estimated function value
        self.gest = np.sum(P / (self.upp - xmma) + Q / (xmma - self.low), axis=1) - rhs
        self.dk = np.sum( (self.upp - self.low) * (xmma - xval)**2 / ((self.upp - xmma) * (xmma - self.low) * self.dx))
        
        return xmma

    @staticmethod
    def subsolv(epsimin, low, upp, alfa, beta, P, Q, a0, a, b, c, d, x0=None):
        r"""This function solves the MMA subproblem

        minimize   f_0(\vec{x}) + a_0*z + \sum_i^m[ c_i*y_i + 1/2*d_i*y_i^2 ],
        subject to f_i(\vec{x}) - a_i*z - y_i <= b_i,   for i = 1, ..., m
                alfa_j <=  x_j <=  beta_j,  for j = 1, ..., n
                y_i >= 0,  for i = 1, ..., m
                z >= 0.

        where:
            MMA approximation: :math:`f_i(\vec{x}) = \sum_j\left( p_{ij}/(upp_j-x_j) + q_{ij}/(x_j-low_j) \right)`
            m: The number of general constraints
            n: The number of variables in :math:`\vec{x}`

        Args:
            epsimin: Solution tolerance on maximum residual
            low: Column vector with the lower asymptotes
            upp: Column vector with the upper asymptotes
            alfa: Vector with the lower bounds for the variables :math:`\vec{x}`
            beta: Vector with the upper bounds for the variables :math:`\vec{x}`
            P: Upper asymptotic amplitudes
            Q: Lower asymptotic amplitudes
            a0: The constants :math:`a_0` in the term :math:`a_0\cdot z`
            a: Vector with the constants :math:`a_i` in the terms :math:`a_i \cdot z`
            c: Vector with the constants :math:`c_i` in the terms :math:`c_i \cdot y_i`
            d: Vector with the constants :math:`d_i` in the terms :math:`0.5 \cdot d_i \cdot y_i^2`
            x0 (optional): Initial guess, in case not given :math:`x_0 = (\alpha + \beta)/2` is used

        Returns:
            x: Vector with the optimal values of the variables :math:`\vec{x}` in the current MMA subproblem
            y: Vector with the optimal values of the variables :math:`y_i` in the current MMA subproblem
            z: Scalar with the optimal value of the variable :math:`z` in the current MMA subproblem
            lam: Lagrange multipliers for the :math:`m` general MMA constraints
            xsi: Lagrange multipliers for the :math:'n' constraints :math:`alfa_j - x_j <= 0`
            eta: Lagrange multipliers for the :math:'n' constraints :math:`x_j - beta_j <= 0`
            mu: Lagrange multipliers for the :math:`m` constraints :math:`-y_i <= 0`
            zet: Lagrange multiplier for the single constraint :math:`-z <= 0`
            s: Slack variables for the m general MMA constraints
        """

        n, m = len(alfa), len(a)
        epsi = 1.0
        maxittt = 400
        x = 0.5 * (alfa + beta) if x0 is None else np.clip(x0, alfa + 1e-10, beta - 1e-10)  # Design variables
        y = np.ones(m)
        z = 1.0
        lam = np.ones(m)
        GG = np.empty((m, n))
        xsi = np.maximum((1.0 / (x - alfa)), 1)
        eta = np.maximum((1.0 / (beta - x)), 1)
        mu = np.maximum(1, 0.5 * c)
        zet = 1.0
        s = np.ones(m)
        bb = np.empty(m + 1)
        AA = np.empty((m + 1, m + 1))

        P0 = np.ascontiguousarray(P[0, :])
        Q0 = np.ascontiguousarray(Q[0, :])
        P1 = np.ascontiguousarray(P[1:, :])
        Q1 = np.ascontiguousarray(Q[1:, :])

        itera = 0
        while epsi > epsimin:
            # main loop + 1
            itera = itera + 1

            # upcoming lines determine the left hand sides, i.e. the resiudals of all constraints
            ux1 = upp - x
            xl1 = x - low

            plam = P0 + np.dot(lam, P1)
            qlam = Q0 + np.dot(lam, Q1)
            gvec = np.dot(P1, 1 / ux1) + np.dot(Q1, 1 / xl1)

            # gradient of approximation function wrt x
            dpsidx = plam / (ux1**2) - qlam / (xl1**2)
            residu = np.concatenate([
                dpsidx - xsi + eta,  # rex [n]
                c + d * y - mu - lam,  # rey [m]
                np.array([a0 - zet - np.dot(a, lam)]),  # rez [1]
                gvec - a * z - y + s - b,  # relam [m]
                xsi * (x - alfa) - epsi,  # rexsi [n]
                eta * (beta - x) - epsi,  # reeta [n]
                mu * y - epsi,  # remu [m]
                np.array([zet * z - epsi]),  # rezet [1]
                lam * s - epsi,  # res [m]
            ])

            resi2 = residu**2
            residunorm = resi2.sum()
            residumax = resi2.max()

            ittt = 0
            # the algorithm is terminated when the maximum residual has become smaller than 0.9*epsilon
            # and epsilon has become sufficiently small (and not too many iterations are used)
            while residumax > (0.9 * epsi)**2 and ittt < maxittt:
                ittt = ittt + 1

                # Newton's method: first create the variable steps

                # precalculations for PSIjj (or diagx)
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1**2
                xl2 = xl1**2

                uxinv1 = 1.0 / ux1
                xlinv1 = 1.0 / xl1
                uxinv2 = 1.0 / ux2
                xlinv2 = 1.0 / xl2

                plam = P0 + np.dot(lam, P1)
                qlam = Q0 + np.dot(lam, Q1)
                gvec = np.dot(P1, uxinv1) + np.dot(Q1, xlinv1)

                # CG is an m x n matrix with values equal to partial derivative of constraints wrt variables
                GG[:, :] = P1 * uxinv2 - Q1 * xlinv2

                # derivative of PSI wrt x
                dpsidx = plam / ux2 - qlam / xl2

                # calculation of right hand sides dx, dy, dz, dlam
                delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
                dely = c + d * y - lam - epsi / y
                delz = a0 - np.dot(a, lam) - epsi / z
                dellam = gvec - a * z - y - b + epsi / lam

                # calculation of diagonal matrices Dx Dy Dlam
                diagx = 2 * (plam / (ux1 * ux2) + qlam / (xl1 * xl2)) + xsi / (x - alfa) + eta / (beta - x)
                diagy = d + mu / y

                diaglam = s / lam
                diaglamyi = diaglam + 1.0 / diagy

                # different options depending on the number of constraints
                # considering the fact I will probably not use local constraints I removed the option

                # normally here is a statement if m < n
                bb[:-1] = dellam + dely / diagy - np.dot(GG, (delx / diagx))
                bb[-1] = delz

                AA[:-1, :-1] = np.diag(diaglamyi) + np.dot((GG / diagx), GG.T)
                AA[-1, :-1] = a
                AA[:-1, -1] = a
                AA[-1, -1] = -zet / z
                # solve system for delta lambda and delta z
                solut = np.linalg.solve(AA, bb)

                # solution of delta vars
                dlam = solut[0:m]
                dz = solut[m]
                dx = -delx / diagx - np.dot(dlam, GG) / diagx
                dy = -dely / diagy + dlam / diagy
                dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
                deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x)
                dmu = -mu + epsi / y - (mu * dy) / y
                dzet = -zet + epsi / z - zet * dz / z
                ds = -s + epsi / lam - (s * dlam) / lam

                # calculate the step size
                stmy = -1.01 * np.min(dy / y)
                stmz = -1.01 * dz / z
                stmlam = -1.01 * np.min(dlam / lam)
                stmxsi = -1.01 * np.min(dxsi / xsi)
                stmeta = -1.01 * np.min(deta / eta)
                stmmu = -1.01 * np.min(dmu / mu)
                stmzet = -1.01 * dzet / zet
                stms = -1.01 * np.min(ds / s)
                stmxx = max(stmy, stmz, stmlam, stmxsi, stmeta, stmmu, stmzet, stms)

                # put variables and accompanying changes in alist
                stmalfa = -1.01 * np.min(dx / (x - alfa))
                stmbeta = 1.01 * np.max(dx / (beta - x))

                # Initial step size
                steg = 1.0 / max(stmalfa, stmbeta, stmxx, 1.0)

                # set old variables
                xold = x.copy()
                yold = y.copy()
                zold = z
                lamold = lam.copy()
                xsiold = xsi.copy()
                etaold = eta.copy()
                muold = mu.copy()
                zetold = zet
                sold = s.copy()

                # Do linesearch
                for itto in range(maxittt):
                    # Find new set of variables with stepsize
                    x[:] = xold + steg * dx
                    y[:] = yold + steg * dy
                    z = zold + steg * dz
                    lam[:] = lamold + steg * dlam
                    xsi[:] = xsiold + steg * dxsi
                    eta[:] = etaold + steg * deta
                    mu[:] = muold + steg * dmu
                    zet = zetold + steg * dzet
                    s[:] = sold + steg * ds

                    # Calculate residual
                    ux1 = upp - x
                    xl1 = x - low

                    plam = P0 + np.dot(lam, P1)
                    qlam = Q0 + np.dot(lam, Q1)
                    gvec = np.dot(P1, 1 / ux1) + np.dot(Q1, 1 / xl1)

                    # gradient of approximation function wrt x
                    dpsidx = plam / (ux1**2) - qlam / (xl1**2)

                    residu = np.concatenate([
                        dpsidx - xsi + eta,  # rex [n]
                        c + d * y - mu - lam,  # rey [m]
                        np.array([a0 - zet - np.dot(a, lam)]),  # rez [1]
                        gvec - a * z - y + s - b,  # relam [m]
                        xsi * (x - alfa) - epsi,  # rexsi [n]
                        eta * (beta - x) - epsi,  # reeta [n]
                        mu * y - epsi,  # remu [m]
                        np.array([zet * z - epsi]),  # rezet [1]
                        lam * s - epsi,  # res [m]
                    ])
                    resi2 = residu**2
                    resinorm_ii = resi2.sum()
                    if resinorm_ii < residunorm:
                        break
                    steg /= 2  # Reduce stepsize

                residunorm = resinorm_ii
                residumax = resi2.max()

            if ittt > maxittt - 2:
                print(f"MMA Subsolver: itt = {ittt}, at epsi = {'%.3e' % epsi}")
            # decrease epsilon with factor 10
            epsi /= 10

        # ## END OF SUBSOLVE
        return x, y, z, lam, xsi, eta, mu, zet, s
