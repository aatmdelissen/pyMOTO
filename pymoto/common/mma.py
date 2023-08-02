import numpy as np
from pymoto.utils import _parse_to_list, _concatenate_to_array


def residual(x, y, z, lam, xsi, eta, mu, zet, s, upp, low, P0, P1, Q0, Q1, epsi, a0, a, b, c, d, alfa, beta):
    # upcoming lines determine the left hand sides, i.e. the resiudals of all constraints
    ux1 = upp - x
    xl1 = x - low

    plam = P0 + np.dot(lam, P1)
    qlam = Q0 + np.dot(lam, Q1)
    gvec = np.dot(P1, 1/ux1) + np.dot(Q1, 1/xl1)

    # gradient of approximation function wrt x
    dpsidx = plam / (ux1**2) - qlam / (xl1**2)

    # put all residuals in one line
    return np.concatenate([
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


def subsolv(epsimin, low, upp, alfa, beta, P, Q, a0, a, b, c, d):
    """ This function subsolv solves the MMA subproblem
    minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
             + SUM[ ci*yi + 0.5*di*(yi)^2 ],
    subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
               alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
    Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.
    """

    n, m = len(alfa), len(a)
    epsi = 1.0
    maxittt = 400
    x = 0.5 * (alfa + beta)
    y = np.ones(m)
    z = 1.0
    lam = np.ones(m)
    GG = np.empty((m, n))
    xsi = np.maximum((1.0 / (x - alfa)), 1)
    eta = np.maximum((1.0 / (beta - x)), 1)
    mu = np.maximum(1, 0.5 * c)
    zet = 1.0
    s = np.ones(m)
    bb = np.empty(m+1)
    AA = np.empty((m+1, m+1))

    P0 = np.ascontiguousarray(P[0, :])
    Q0 = np.ascontiguousarray(Q[0, :])
    P1 = np.ascontiguousarray(P[1:, :])
    Q1 = np.ascontiguousarray(Q[1:, :])

    itera = 0
    while epsi > epsimin:
        # main loop + 1
        itera = itera + 1

        # upcoming lines determine the left hand sides, i.e. the resiudals of all constraints
        residu = residual(x, y, z, lam, xsi, eta, mu, zet, s, upp, low, P0, P1, Q0, Q1, epsi, a0, a, b, c, d, alfa, beta)
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
            ux2 = ux1 ** 2
            xl2 = xl1 ** 2
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
            AA[-1, -1] = -zet/z
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
            stmy = -1.01*np.min(dy/y)
            stmz = -1.01 * dz / z
            stmlam = -1.01*np.min(dlam / lam)
            stmxsi = -1.01*np.min(dxsi / xsi)
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
            itto = 0
            while itto < maxittt:
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

                residu = residual(x, y, z, lam, xsi, eta, mu, zet, s, upp, low, P0, P1, Q0, Q1, epsi, a0, a, b, c, d, alfa, beta)
                if np.linalg.norm(residu) < residunorm:
                    break
                itto += 1
                steg /= 2  # Reduce stepsize

            residunorm = np.linalg.norm(residu)
            residumax = np.max(np.abs(residu))

        if ittt > maxittt - 2:
            print(f"MMA Subsolver: itt = {ittt}, at epsi = {epsi}")
        # decrease epsilon with factor 10
        epsi /= 10

    # ## END OF SUBSOLVE
    return x, y, z, lam, xsi, eta, mu, zet, s


class MMA:
    """
    Block for the MMA algorithm
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
        verbosity: 0 - No prints, 1 - Only convergence message, 2 - Convergence and iteration info, 3 - Extended info

    """

    def __init__(self, function, variables, responses, tolx=1e-4, tolf=0.0, move=0.1, maxit=100, xmin=0.0, xmax=1.0, fn_callback=None, verbosity=0, **kwargs):
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

        self.epsimin = kwargs.get("epsimin", 1e-7)  # Or 1e-7 ?? witout sqrt(m+n) or 1e-9
        self.raa0 = kwargs.get("raa0", 1e-5)

        self.cCoef = kwargs.get("cCoef", 1e3)  # Svanberg uses 1e3 in example? Old code had 1e7

        # Not used
        self.dxmin = kwargs.get("dxmin", 1e-5)

        self.albefa = kwargs.get("albefa", 0.1)
        self.asyinit = kwargs.get("asyinit", 0.5)
        self.asyincr = kwargs.get("asyincr", 1.2)
        self.asydecr = kwargs.get("asydecr", 0.7)
        self.asybound = kwargs.get("asybound", 10.0)

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
        self.a = np.zeros(self.m)
        self.c = self.cCoef * np.ones(self.m)
        self.d = np.ones(self.m)
        self.gold1 = np.zeros(self.m + 1)
        self.gold2 = self.gold1.copy()
        self.rho = self.raa0 * np.ones(self.m + 1)

    def response(self):
        change = 1

        # Save initial state
        xval, self.cumlens = _concatenate_to_array([s.state for s in self.variables])
        self.n = len(xval)

        # Set outer bounds
        if not hasattr(self.xmin, '__len__'):
            self.xmin = self.xmin * np.ones_like(xval)
        elif len(self.xmin) == len(self.variables):
            xminvals = self.xmin
            self.xmin = np.zeros_like(xval)
            for i in range(len(xminvals)):
                self.xmin[self.cumlens[i]:self.cumlens[i+1]] = xminvals[i]

        if len(self.xmin) != self.n:
            raise RuntimeError(
                "Length of the xmin vector not correct ({} != {})".format(len(self.xmin), self.n))

        if not hasattr(self.xmax, '__len__'):
            self.xmax = self.xmax * np.ones_like(xval)
        elif len(self.xmax) == len(self.variables):
            xmaxvals = self.xmax
            self.xmax = np.zeros_like(xval)
            for i in range(len(xmaxvals)):
                self.xmax[self.cumlens[i]:self.cumlens[i + 1]] = xmaxvals[i]

        if len(self.xmax) != self.n:
            raise RuntimeError("Length of the xmax vector not correct ({} != {})".format(len(self.xmax), self.n))

        # Set movelimit in case of multiple
        if hasattr(self.move, '__len__'):
            if len(self.move) == len(self.variables):
                movevals = self.move
                self.move = np.zeros_like(xval)
                for i in range(len(movevals)):
                    self.move[self.cumlens[i]:self.cumlens[i + 1]] = movevals[i]
            elif len(self.move) != self.n:
                raise RuntimeError("Length of the move vector not correct ({} != {})".format(len(self.move), self.n))

        fcur = 0.0
        while self.iter < self.maxIt:
            # Reset all signals in function block
            self.funbl.reset()

            # Set the new states
            for i, s in enumerate(self.variables):
                if self.cumlens[i+1]-self.cumlens[i] == 1:
                    try:
                        s.state[:] = xval[self.cumlens[i]]
                    except TypeError:
                        s.state = xval[self.cumlens[i]]
                else:
                    s.state[:] = xval[self.cumlens[i]:self.cumlens[i+1]]

            if self.fn_callback is not None:
                self.fn_callback()

            # Calculate response
            self.funbl.response()

            # Update the states
            for i, s in enumerate(self.variables):
                if self.cumlens[i+1]-self.cumlens[i] == 1:
                    try:
                        xval[self.cumlens[i]] = s.state[:]
                    except (TypeError, IndexError):
                        xval[self.cumlens[i]] = s.state
                else:
                    xval[self.cumlens[i]:self.cumlens[i+1]] = s.state[:]

            # Save response
            f = ()
            for s in self.responses:
                f += (s.state, )

            # Check function change convergence criterion
            fprev, fcur = fcur, self.responses[0].state
            rel_fchange = abs(fcur-fprev)/abs(fcur)
            if rel_fchange < self.tolf:
                if self.verbosity >= 1:
                    print(f"MMA converged: Relative function change |Δf|/|f| ({rel_fchange}) below tolerance ({self.tolf})")
                break

            # Calculate and save sensitivities
            df = ()
            for i, s_out in enumerate(self.responses):
                for s in self.responses:
                    s.reset()

                s_out.sensitivity = s_out.state*0 + 1.0

                self.funbl.sensitivity()

                sens_list = []
                for v in self.variables:
                    sens_list.append(v.sensitivity if v.sensitivity is not None else 0*v.state)
                dff, _ = _concatenate_to_array(sens_list)
                df += (dff, )

                # Reset sensitivities for the next response
                self.funbl.reset()

            # Display info on variables
            if self.verbosity >= 3:
                for i, s in enumerate(self.variables):
                    isscal = self.cumlens[i + 1] - self.cumlens[i] == 1
                    msg = "{0:>10s} = ".format(s.tag)
                    if isscal:
                        try:
                            msg += "         {0: .3e}         ".format(s.state)
                        except TypeError:
                            msg += "         {0: .3e}         ".format(s.state[0])
                    else:
                        msg += "[{0: .3e} ... {1: .3e}] ".format(min(s.state), max(s.state))
                    for j, s_out in enumerate(self.responses):
                        msg += "| {0:>10s}/{1:10s} = ".format("d"+s_out.tag, "d"+s.tag)
                        if isscal:
                            msg += "         {0: .3e}         ".format(df[j][self.cumlens[i]])
                        else:
                            msg += "[{0: .3e} ... {1: .3e}] ".format(min(df[j][self.cumlens[i]:self.cumlens[i+1]]), max(df[j][self.cumlens[i]:self.cumlens[i+1]]))
                    print(msg)

            self.iter += 1
            xnew, change = self.mmasub(xval.copy(), np.hstack(f), np.vstack(df))

            # Stopping criteria on step size
            rel_stepsize = np.linalg.norm((xval - xnew)/self.dx) / np.linalg.norm(xval/self.dx)
            if rel_stepsize < self.tolX:
                if self.verbosity >= 1:
                    print(f"MMA converged: Relative stepsize |Δx|/|x| ({rel_stepsize}) below tolerance ({self.tolX})")
                break

            xval = xnew

    def mmasub(self, xval, g, dg):
        if self.dx is None:
            self.dx = self.xmax - self.xmin
        if self.offset is None:
            self.offset = self.asyinit * np.ones(self.n)

        #      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
        #    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
        #                xmin_j <= x_j <= xmax_j,    j = 1,...,n
        #                z >= 0,   y_i >= 0,         i = 1,...,m
        # *** INPUT:
        #
        #   m    = The number of general constraints.
        #   n    = The number of variables x_j.
        #  iter  = Current iteration number ( =1 the first time mmasub is called).
        #  xval  = Column vector with the current values of the variables x_j.
        #  xmin  = Column vector with the lower bounds for the variables x_j.
        #  xmax  = Column vector with the upper bounds for the variables x_j.
        #  xold1 = xval, one iteration ago (provided that iter>1).
        #  xold2 = xval, two iterations ago (provided that iter>2).
        #  f0val = The value of the objective function f_0 at xval.
        #  df0dx = Column vector with the derivatives of the objective function
        #          f_0 with respect to the variables x_j, calculated at xval.
        #  fval  = Column vector with the values of the constraint functions f_i,
        #          calculated at xval.
        #  dfdx  = (m x n)-matrix with the derivatives of the constraint functions
        #          f_i with respect to the variables x_j, calculated at xval.
        #          dfdx(i,j) = the derivative of f_i with respect to x_j.
        #  low   = Column vector with the lower asymptotes from the previous
        #          iteration (provided that iter>1).
        #  upp   = Column vector with the upper asymptotes from the previous
        #          iteration (provided that iter>1).
        #  a0    = The constants a_0 in the term a_0*z.
        #  a     = Column vector with the constants a_i in the terms a_i*z.
        #  c     = Column vector with the constants c_i in the terms c_i*y_i.
        #  d     = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
        #

        # *** OUTPUT:
        #
        #  xmma  = Column vector with the optimal values of the variables x_j
        #          in the current MMA subproblem.
        #  ymma  = Column vector with the optimal values of the variables y_i
        #          in the current MMA subproblem.
        #  zmma  = Scalar with the optimal value of the variable z
        #          in the current MMA subproblem.
        #  lam   = Lagrange multipliers for the m general MMA constraints.
        #  xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
        #  eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
        #   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
        #  zet   = Lagrange multiplier for the single constraint -z <= 0.
        #   s    = Slack variables for the m general MMA constraints.
        #  low   = Column vector with the lower asymptotes, calculated and used
        #          in the current MMA subproblem.
        #  upp   = Column vector with the upper asymptotes, calculated and used
        #          in the current MMA subproblem.

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
            self.offset = np.clip(self.offset, 1/(self.asybound**2), self.asybound)

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
        dx2 = shift**2
        P = dx2 * np.maximum(+dg, 0)
        Q = dx2 * np.maximum(-dg, 0)

        rhs = np.dot(P, 1 / shift) + np.dot(Q, 1 / shift) - g
        b = rhs[1:]

        # Solving the subproblem by a primal-dual Newton method
        epsimin_scaled = self.epsimin*np.sqrt(self.m + self.n)
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(epsimin_scaled, self.low, self.upp, alfa, beta, P, Q, self.a0, self.a, b, self.c, self.d)

        self.gold2, self.gold1 = self.gold1, g.copy()
        self.xold2, self.xold1 = self.xold1, xval.copy()
        change = np.average(abs(xval - xmma))

        if self.verbosity >= 2:
            msgs = ["g{0:d}({1:s}): {2:+.4e}".format(i, s.tag, g[i]) for i, s in enumerate(self.responses)]
            print("It. {0: 4d}, {1}".format(self.iter, ", ".join(msgs)))

        if self.verbosity >=3:
            # Print changes
            printstr = "Changes: "
            for i, s in enumerate(self.variables):
                isscal = self.cumlens[i + 1] - self.cumlens[i] == 1
                if isscal:
                    chg = abs(xval[self.cumlens[i]] - xmma[self.cumlens[i]])
                else:
                    chg = np.average(abs(xval[self.cumlens[i]:self.cumlens[i + 1]] - xmma[self.cumlens[i]:self.cumlens[i + 1]]))

                printstr += "{0:s} = {1:.3e}   ".format("Δ_"+s.tag, chg)
            print(printstr)

        return xmma, change