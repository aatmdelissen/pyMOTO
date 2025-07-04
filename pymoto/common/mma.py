import numpy as np
from pymoto.utils import _parse_to_list, _concatenate_to_array


def residual(x, y, z, lam, xsi, eta, mu, zet, s, upp, low, P0, P1, Q0, Q1, epsi, a0, a, b, c, d, alfa, beta):
    # upcoming lines determine the left hand sides, i.e. the resiudals of all constraints
    ux1 = upp - x
    xl1 = x - low

    plam = P0 + np.dot(lam, P1)
    qlam = Q0 + np.dot(lam, Q1)
    gvec = np.dot(P1, 1 / ux1) + np.dot(Q1, 1 / xl1)

    # gradient of approximation function wrt x
    dpsidx = plam / (ux1**2) - qlam / (xl1**2)

    # put all residuals in one line
    return np.concatenate(
        [
            dpsidx - xsi + eta,  # rex [n]
            c + d * y - mu - lam,  # rey [m]
            np.array([a0 - zet - np.dot(a, lam)]),  # rez [1]
            gvec - a * z - y + s - b,  # relam [m]
            xsi * (x - alfa) - epsi,  # rexsi [n]
            eta * (beta - x) - epsi,  # reeta [n]
            mu * y - epsi,  # remu [m]
            np.array([zet * z - epsi]),  # rezet [1]
            lam * s - epsi,  # res [m]
        ]
    )


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
        residu = residual(
            x, y, z, lam, xsi, eta, mu, zet, s, upp, low, P0, P1, Q0, Q1, epsi, a0, a, b, c, d, alfa, beta
        )
        residunorm = np.linalg.norm(residu)
        residumax = np.max(np.abs(residu))

        ittt = 0
        # the algorithm is terminated when the maximum residual has become smaller than 0.9*epsilon
        # and epsilon has become sufficiently small (and not too many iterations are used)
        while residumax > 0.9 * epsi and ittt < maxittt:
            ittt = ittt + 1

            # Newton's method: first create the variable steps

            # precalculations for PSIjj (or diagx)
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1**2
            xl2 = xl1**2
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2

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
            diagx = 2 * (plam / ux3 + qlam / xl3) + xsi / (x - alfa) + eta / (beta - x)
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

                residu = residual(
                    x, y, z, lam, xsi, eta, mu, zet, s, upp, low, P0, P1, Q0, Q1, epsi, a0, a, b, c, d, alfa, beta
                )
                if np.linalg.norm(residu) < residunorm:
                    break
                steg /= 2  # Reduce stepsize

            residunorm = np.linalg.norm(residu)
            residumax = np.max(np.abs(residu))

        if ittt > maxittt - 2:
            print(f"MMA Subsolver: itt = {ittt}, at epsi = {'%.3e' % epsi}")
        # decrease epsilon with factor 10
        epsi /= 10

    # ## END OF SUBSOLVE
    return x, y, z, lam, xsi, eta, mu, zet, s


class MMA:
    r"""Class for the MMA optimization algorithm
    The design variables are set by keyword <variables> accepting a list of variables.
    The responses are set by keyword <responses> accepting a list of signals.
    If none are given, the internal sig_in and sig_out are used.

    Args:
        function: The Network defining the optimization problem
        variables: The Signals defining the design variables
        responses: A list of Signals, where the first is to be minimized and the others are constraints.

    Keyword Args:
        tolx: Stopping criterium for relative design change
        tolf: Stopping criterium for relative objective change
        maxit: Maximum number of iteration
        move: Move limit on relative variable change per iteration
        xmin: Minimum design variable (can be a vector)
        xmax: Maximum design variable (can be a vector)
        fn_callback: A function that is called just before calling the response() in each iteration
        verbosity: Level of information to print
          0 - No prints
          1 - Only convergence message
          2 - Convergence and iteration info (default)
          3 - Additional info on variables
          4 - Additional info on sensitivity information

    """

    def __init__(
        self,
        function,
        variables,
        responses,
        tolx=1e-4,
        tolf=0.0,
        move=0.1,
        maxit=100,
        xmin=0.0,
        xmax=1.0,
        fn_callback=None,
        verbosity=2,
        **kwargs,
    ):
        self.funbl = function
        self.verbosity = verbosity

        self.variables = _parse_to_list(variables)
        self.responses = _parse_to_list(responses)

        self.iter = 0

        # Convergence options
        self.tolX = tolx
        self.tolf = tolf
        self.maxIt = maxit

        # Operational options
        self.xmax = xmax
        self.xmin = xmin
        self.move = move

        self.pijconst = kwargs.get("pijconst", 1e-3)

        self.a0 = kwargs.get("a0", 1.0)

        self.epsimin = kwargs.get("epsimin", 1e-10)  # Or 1e-7 ?? witout sqrt(m+n) or 1e-9
        self.cCoef = kwargs.get("cCoef", 1e3)  # Svanberg uses 1e3 in example? Old code had 1e7

        self.albefa = kwargs.get("albefa", 0.1)
        self.asyinit = kwargs.get("asyinit", 0.5)
        self.asyincr = kwargs.get("asyincr", 1.2)
        self.asydecr = kwargs.get("asydecr", 0.7)
        self.asybound = kwargs.get("asybound", 10.0)
        self.mmaversion = kwargs.get("mmaversion", "Svanberg2007")  # Options are Svanberg1987, Svanberg2007

        self.ittomax = kwargs.get("ittomax", 400)

        self.iterinitial = kwargs.get("iterinitial", 2.5)

        self.fn_callback = fn_callback

        # Numbers
        self.n = None  # len(x0)
        self.dx = None
        self.xold1 = None
        self.xold2 = None
        self.low = None
        self.upp = None
        self.offset = None

        # Setting up for constriants
        self.m = len(self.responses) - 1
        self.a = kwargs.get("a", np.zeros(self.m))
        if len(self.a) != self.m:
            raise RuntimeError(f"Length of the a vector ({len(self.a)}) should be equal to # constraints ({self.m}).")
        self.c = kwargs.get("c", np.full(self.m, self.cCoef, dtype=float))
        if len(self.c) != self.m:
            raise RuntimeError(f"Length of the c vector ({len(self.c)}) should be equal to # constraints ({self.m}).")
        self.d = np.ones(self.m)
        self.gold1 = np.zeros(self.m + 1)
        self.gold2 = self.gold1.copy()

    def response(self):
        change = 1

        # Save initial state
        xval, self.cumlens = _concatenate_to_array([s.state for s in self.variables])
        self.n = len(xval)

        # Set outer bounds
        if not hasattr(self.xmin, "__len__"):
            self.xmin = self.xmin * np.ones_like(xval)
        elif len(self.xmin) == len(self.variables):
            xminvals = self.xmin
            self.xmin = np.zeros_like(xval)
            for i in range(len(xminvals)):
                self.xmin[self.cumlens[i] : self.cumlens[i + 1]] = xminvals[i]

        if len(self.xmin) != self.n:
            raise RuntimeError(
                f"Length of the xmin vector ({len(self.xmin)}) should be equal to # design variables ({self.n})"
            )

        if not hasattr(self.xmax, "__len__"):
            self.xmax = self.xmax * np.ones_like(xval)
        elif len(self.xmax) == len(self.variables):
            xmaxvals = self.xmax
            self.xmax = np.zeros_like(xval)
            for i in range(len(xmaxvals)):
                self.xmax[self.cumlens[i] : self.cumlens[i + 1]] = xmaxvals[i]

        if len(self.xmax) != self.n:
            raise RuntimeError(
                f"Length of the xmax vector ({len(self.xmax)}) should be equal to # design variables ({self.n})"
            )

        if hasattr(self.move, "__len__"):
            # Set movelimit in case of multiple are given
            move_input = np.asarray(self.move).copy()
            if move_input.size == len(self.variables):
                self.move = np.zeros_like(xval)
                for i in range(move_input.size):
                    self.move[self.cumlens[i] : self.cumlens[i + 1]] = move_input[i]
            elif len(self.move) != self.n:
                raise RuntimeError(
                    f"Length of the move vector ({len(self.move)}) should be equal to number of "
                    f"design variable signals ({len(self.variables)}) or "
                    f"total number of design variables ({self.n})."
                )

        fcur = 0.0
        while self.iter < self.maxIt:
            # Reset all signals in function block
            self.funbl.reset()

            # Set the new states
            for i, s in enumerate(self.variables):
                if self.cumlens[i + 1] - self.cumlens[i] == 1:
                    s.state = xval[self.cumlens[i]]
                else:
                    s.state = xval[self.cumlens[i] : self.cumlens[i + 1]]

            if self.fn_callback is not None:
                self.fn_callback()

            # Calculate response
            self.funbl.response()

            xval, _ = _concatenate_to_array([s.state for s in self.variables])

            # Save response
            f = ()
            for s in self.responses:
                if np.size(s.state) != 1:
                    raise TypeError("State of responses must be scalar.")
                if np.iscomplexobj(s.state):
                    raise TypeError("Responses must be real-valued.")
                f += (s.state,)

            # Check function change convergence criterion
            fprev, fcur = fcur, self.responses[0].state
            rel_fchange = abs(fcur - fprev) / abs(fcur)
            if rel_fchange < self.tolf:
                if self.verbosity >= 1:
                    print(
                        (
                            "MMA converged: Relative function change |Δf|/|f| ",
                            f"({rel_fchange}) below tolerance ({self.tolf})",
                        )
                    )
                break

            # Calculate and save sensitivities
            df = ()
            for i, s_out in enumerate(self.responses):
                for s in self.responses:
                    s.reset()

                s_out.sensitivity = s_out.state * 0 + 1.0

                self.funbl.sensitivity()

                sens_list = []
                for v in self.variables:
                    sens_list.append(v.sensitivity if v.sensitivity is not None else 0 * v.state)
                dff, _ = _concatenate_to_array(sens_list)
                df += (dff,)

                # Reset sensitivities for the next response
                self.funbl.reset()

            if self.verbosity >= 3:
                # Display info on variables
                show_sensitivities = self.verbosity >= 4
                msg = ""
                for i, s in enumerate(self.variables):
                    if show_sensitivities:
                        msg += "{0:>10s} = ".format(s.tag[:10])
                    else:
                        msg += f"{s.tag} = "

                    # Display value range
                    fmt = "% .2e"
                    minval, maxval = np.min(s.state), np.max(s.state)
                    mintag, maxtag = fmt % minval, fmt % maxval
                    if mintag == maxtag:
                        if show_sensitivities:
                            msg += f"       {mintag}      "
                        else:
                            msg += f" {mintag}"
                    else:
                        sep = "…" if len(s.state) > 2 else ","
                        msg += f"[{mintag}{sep}{maxtag}]"
                        if show_sensitivities:
                            msg += " "

                    if show_sensitivities:
                        # Display info on sensivity values
                        for j, s_out in enumerate(self.responses):
                            msg += "| {0:s}/{1:11s} = ".format("d" + s_out.tag, "d" + s.tag[:10])
                            minval = np.min(df[j][self.cumlens[i] : self.cumlens[i + 1]])
                            maxval = np.max(df[j][self.cumlens[i] : self.cumlens[i + 1]])
                            mintag, maxtag = fmt % minval, fmt % maxval
                            if mintag == maxtag:
                                msg += f"       {mintag}      "
                            else:
                                sep = "…" if self.cumlens[i + 1] - self.cumlens[i] > 2 else ","
                                msg += f"[{mintag}{sep}{maxtag}] "
                        msg += "\n"
                    elif i != len(self.variables) - 1:
                        msg += ", "
                print(msg)

            xnew, change = self.mmasub(xval.copy(), np.hstack(f), np.vstack(df))

            # Stopping criteria on step size
            rel_stepsize = np.linalg.norm((xval - xnew) / self.dx) / np.linalg.norm(xval / self.dx)
            if rel_stepsize < self.tolX:
                if self.verbosity >= 1:
                    print(f"MMA converged: Relative stepsize |Δx|/|x| ({rel_stepsize}) below tolerance ({self.tolX})")
                break

            xval = xnew
            self.iter += 1

    def mmasub(self, xval, g, dg):
        if self.dx is None:
            self.dx = self.xmax - self.xmin
        if self.offset is None:
            self.offset = self.asyinit * np.ones(self.n)

        # # ASYMPTOTES
        # Calculation of the asymptotes low and upp :
        # For iter = 1,2 the asymptotes are fixed depending on asyinit
        if self.xold1 is not None and self.xold2 is not None:
            # depending on if the signs of xval - xold and xold - xold2 are opposite, indicating an oscillation
            # in the variable xi
            # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed

            # check for oscillations in variables
            # if zzz positive no oscillations, if negative --> oscillations
            zzz = (xval - self.xold1) * (self.xold1 - self.xold2)
            # decrease those variables that are oscillating equal to asydecr
            self.offset[zzz > 0] *= self.asyincr
            self.offset[zzz < 0] *= self.asydecr

            # check with minimum and maximum bounds of asymptotes, as they cannot be to close or far from the variable
            # give boundaries for upper and lower asymptotes
            self.offset = np.clip(self.offset, 1 / (self.asybound**2), self.asybound)

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
        if "1987" in self.mmaversion:
            # Original version
            P = dx2 * dg_plus
            Q = dx2 * dg_min
        elif "2007" in self.mmaversion:
            # Improved version -> Allows to use higher epsimin to get design variables closer to the bound.
            P = dx2 * (1.001 * dg_plus + 0.001 * dg_min + 1e-5 / self.dx)
            Q = dx2 * (0.001 * dg_plus + 1.001 * dg_min + 1e-5 / self.dx)
        else:
            raise ValueError('Only "Svanberg1987" or "Svanberg2007" are valid options')

        rhs = np.dot(P, 1 / shift) + np.dot(Q, 1 / shift) - g
        b = rhs[1:]

        # Solving the subproblem by a primal-dual Newton method
        epsimin_scaled = self.epsimin * np.sqrt(self.m + self.n)
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
            epsimin_scaled, self.low, self.upp, alfa, beta, P, Q, self.a0, self.a, b, self.c, self.d, x0=xval
        )

        self.gold2, self.gold1 = self.gold1, g.copy()
        self.xold2, self.xold1 = self.xold1, xval.copy()
        change = np.average(abs(xval - xmma))

        if self.verbosity >= 2:
            # Display iteration status message
            msgs = ["g{0:d}({1:s}): {2:+.4e}".format(i, s.tag, g[i]) for i, s in enumerate(self.responses)]
            max_infeasibility = max(g[1:])
            is_feasible = max_infeasibility <= 0

            feasibility_tag = "f" if is_feasible else " "
            print("It. {0: 4d}, [{1:1s}] {2}".format(self.iter, feasibility_tag, ", ".join(msgs)))

        if self.verbosity >= 3:
            # Report design feasibility
            iconst_max = np.argmax(g[1:])
            print(
                f"  | {np.sum(g[1:] > 0)} / {len(g) - 1} violated constraints, "
                f"max. violation ({self.responses[iconst_max + 1].tag}) = {'%.2g' % g[iconst_max + 1]}"
            )

            # Print design changes
            change_msgs = []
            for i, s in enumerate(self.variables):
                minchg = np.min(
                    abs(xval[self.cumlens[i] : self.cumlens[i + 1]] - xmma[self.cumlens[i] : self.cumlens[i + 1]])
                )
                maxchg = np.max(
                    abs(xval[self.cumlens[i] : self.cumlens[i + 1]] - xmma[self.cumlens[i] : self.cumlens[i + 1]])
                )
                fmt = "%.2g"
                mintag, maxtag = fmt % minchg, fmt % maxchg

                if mintag == maxtag:
                    change_msgs.append(f"Δ({s.tag}) = {mintag}")
                else:
                    change_msgs.append(f"Δ({s.tag}) = {mintag}…{maxtag}")

            print(f"  | Changes: {', '.join(change_msgs)}")

        return xmma, change
