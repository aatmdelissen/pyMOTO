from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.optimize import linprog
from ..utils import _parse_to_list, _concatenate_to_array
from ..core_objects import Network, Module, SignalsT, Signal


class Optimizer(ABC):
    """General abstract optimizer object"""
    def __init__(self, 
                 variables: SignalsT, 
                 responses: SignalsT, 
                 function: Network = None, 
                 slice_network: bool = False, 
                 xmin=None, 
                 xmax=None,         
                 verbosity: int = 2,
    ):
        """Initialize general optimization object

        Args:
            variables (Signal(s)): One or more variable Signals defining the design variables
            responses (Signal(s)): One or more response Signals, where the first is to be minimized and the others are 
              constraints in negative null form.
            function (Network, optional): The Network defining the optimization problem. Defaults to None.
            slice_network (bool, optional): If True, only the modules connecting variable and response signals are
              evaluated. Defaults to False.
            xmin (optional): Minimum design variable (can be passed as scalar: same value for all variables, 
              vector: each variable has a unique value, list of scalars/vectors: for each variable signal)
            xmax (optional): Maximum design variable (can be passed as scalar: same value for all variables, 
              vector: each variable has a unique value, list of scalars/vectors: for each variable signal)
            verbosity (int, optional): Level of information to printDefaults to 2.
              0 - No prints
              1 - Only convergence message
              2 - Convergence and iteration info (default)
              3 - Additional info on variables and GCMMA inner iteration info (when applicable)
              4 - Additional info on sensitivity information. 
        """
        self.variables = _parse_to_list(variables)
        self.responses = _parse_to_list(responses)

        self.verbosity = verbosity

        if function is None:
            function = Network.active[0]

        if slice_network:
            subfn = function.get_output_cone(self.responses).get_input_cone(self.variables)
            if len(subfn) == 0:
                raise RuntimeError(
                    f"Could not find a network that uses the provided input signals {self.responses} "
                    f"and produces the requested output signals {self.variables}"
                )

            # Check if all required states have a value, else try to run anything up to input signals
            for s in self.variables:
                if s.state is None:
                    function.get_output_cone(tosig=s).response()
                if s.state is None:
                    raise RuntimeError(f"Input signal {s} has no state.")

            function = subfn

        self.function = function

        self.response_is_uptodate = not any([s.state is None for s in self.responses])

        # Convert variables to a vector
        xval, self._cumlens = _concatenate_to_array([s.state for s in self.variables])
        self.n = xval.size  # Number of design parameters

        # Set lower bound
        if xmin is None and all([hasattr(s, 'min') for s in self.variables]):
            # Try to obtain from signal
            xmin = [s.min for s in self.variables]

        if xmin is not None:
            self.xmin = self._parse_bound(xmin, which='xmin')
        
        # Set upper bound
        if xmax is None and all([hasattr(s, 'max') for s in self.variables]):
            # Try to obtain from signal
            xmax = [s.max for s in self.variables]

        if xmax is not None:
            self.xmax = self._parse_bound(xmax, which='xmax')

        # Initialize other parameters
        self.iter = 0

    def _parse_bound(self, xbnd, which='bounds'):
        """ Helper function to get upper and lower bound vector"""
        xbnd = np.asarray(xbnd)
        if xbnd.size == 1:  # Just one value is given
            bvec = xbnd * np.ones(self.n)  # TODO: Check if this can be replaced with one value
        elif xbnd.size == len(self.variables):  # One value for each signal
            bvec = np.zeros(self.n)
            for i in range(len(xbnd)):
                bvec[self._cumlens[i] : self._cumlens[i + 1]] = xbnd[i]
        elif xbnd.size == self.n:
            bvec = xbnd
        else:
            raise RuntimeError(
                f"""Size of {which} ({xbnd.size}) should be either: 
                    - scalar
                    - equal to the number of variable signals ({len(self.variables)})
                    - equal to number of design variables ({self.n})
                """)
                    
        assert bvec.size == self.n
        return bvec

    @property
    def x(self):
        """ Set current design variable vector """
        xval, _ = _concatenate_to_array([s.state for s in self.variables])
        assert xval.size == self.n
        return xval

    @x.setter
    def x(self, v):
        # Set the new states
        for i, s in enumerate(self.variables):
            dx = np.linalg.norm(s.state - v[self._cumlens[i] : self._cumlens[i + 1]])
            if dx != 0:
                self.response_is_uptodate = False
            if self._cumlens[i + 1] - self._cumlens[i] == 1:
                s.state = v[self._cumlens[i]]  # Don't use slice here because we want a scalar
            else:
                s.state = v[self._cumlens[i] : self._cumlens[i + 1]]

    def calculate_g(self):
        """ Calculate function response g(x) """
        if not self.response_is_uptodate:
            self.function.response()  # Automatically calculate response when x is out of date
            self.response_is_uptodate = True
        # Get responses
        if any(s.state is None for s in self.responses):
            raise ValueError("Response is `None` and may not yet been calculated.")
        if any(np.iscomplexobj(s.state) for s in self.responses):
            raise TypeError("Responses for optimization must be real-valued.")
        if any(np.asarray(s.state).size > 1 for s in self.responses):
            raise TypeError("Responses for optimziation must be scalar.")  # Else calculate_dg must be adapted
        return _concatenate_to_array([s.state for s in self.responses])[0]
    
    def calculate_dg(self):
        """ Calculate Jacobian dg/dx """
        dg = ()
        self.function.reset()
        for i, s_out in enumerate(self.responses):
            s_out.sensitivity = s_out.state * 0 + 1.0   # Seed response

            self.function.sensitivity()  # Backpropagation

            sens_list = [v.sensitivity if v.sensitivity is not None else 0 * v.state for v in self.variables]
            dg += (_concatenate_to_array(sens_list)[0],)

            self.function.reset()  # Reset sensitivities for the next sensitivity
        return np.vstack(dg)
    
    def print_variable_info(self, dg: np.ndarray = None):
        """Print information on variables (and sensitivities)

        Args:
            dg (np.ndarray, optional): If provided, information on sensitivities will also be printed. Defaults to None.
        """
        show_sensitivity_info = dg is not None
        msg = ""
        for i, s in enumerate(self.variables):
            if show_sensitivity_info:
                msg += "{0:>10s} = ".format(s.tag[:10])
            else:
                msg += f"{s.tag} = "

            # Display value range
            fmt = "% .2e"
            minval, maxval = np.min(s.state), np.max(s.state)
            mintag, maxtag = fmt % minval, fmt % maxval
            if mintag == maxtag:
                if show_sensitivity_info:
                    msg += f"       {mintag}      "
                else:
                    msg += f" {mintag}"
            else:
                sep = "…" if len(s.state) > 2 else ","
                msg += f"[{mintag}{sep}{maxtag}]"
                if show_sensitivity_info:
                    msg += " "

            if show_sensitivity_info:
                # Display info on sensivity values
                for j, s_out in enumerate(self.responses):
                    msg += "| {0:s}/{1:11s} = ".format("d" + s_out.tag, "d" + s.tag[:10])
                    minval = np.min(dg[j, self._cumlens[i] : self._cumlens[i + 1]])
                    maxval = np.max(dg[j, self._cumlens[i] : self._cumlens[i + 1]])
                    mintag, maxtag = fmt % minval, fmt % maxval
                    if mintag == maxtag:
                        msg += f"       {mintag}      "
                    else:
                        sep = "…" if self._cumlens[i + 1] - self._cumlens[i] > 2 else ","
                        msg += f"[{mintag}{sep}{maxtag}] "
                if i != len(self.variables) - 1:
                    msg += "\n"
            elif i != len(self.variables) - 1:
                msg += ", "
        print(msg)

    def print_iteration_info(self, g: np.ndarray, report_feasibility: bool = False, 
                             xold: np.ndarray = None, xnew: np.ndarray = None):
        """Print iteration information

        Args:
            g (np.ndarray): Response values
            report_feasibility (bool, optional): Print # violated constraints and worst value. Defaults to False.
            xold (np.ndarray, optional): If provided, shows information on design change. Defaults to None.
            xnew (np.ndarray, optional): If provided, shows information on design change. Defaults to None.
        """
        nresp = len(self.responses)
        # Display iteration status message
        msgs = ["g{0:d}({1:s}): {2:+.4e}".format(i, s.tag, g[i]) for i, s in enumerate(self.responses)]
        if nresp > 1:
            max_infeasibility = max(g[1:])
            is_feasible = max_infeasibility <= 0
            feasibility_tag = "[f] " if is_feasible else "[ ] "
        else:
            feasibility_tag = ""  # No constraints, so always feasible :)

        print("It. {0: 4d}, {1:s}{2}".format(self.iter, feasibility_tag, ", ".join(msgs)))

        if report_feasibility:
            if g[1:].size > 0:
                iconst_max = np.argmax(g[1:])
                print(
                    f"  | {np.sum(g[1:] > 0)} / {nresp - 1} violated constraints, "
                    f"max. violation ({self.responses[iconst_max + 1].tag}) = {'%.2g' % g[iconst_max + 1]}"
                )
        
        if xnew is not None and xold is not None:
            x_diff = np.abs(xnew - xold)
            change_msgs = []
            for i, s in enumerate(self.variables):
                minchg = np.min(x_diff[self._cumlens[i] : self._cumlens[i + 1]])
                maxchg = np.max(x_diff[self._cumlens[i] : self._cumlens[i + 1]])
                fmt = "%.2g"
                mintag, maxtag = fmt % minchg, fmt % maxchg

                if mintag == maxtag:
                    change_msgs.append(f"Δ({s.tag}) = {mintag}")
                else:
                    change_msgs.append(f"Δ({s.tag}) = {mintag}…{maxtag}")

            print(f"  | Changes: {', '.join(change_msgs)}")
    
    def optimize(self, maxiter: int = 100, tolx: float = 1e-4, tolf: float = 1e-4):
        """ Perform a gradient-based optimization 

        Args:
            maxiter (int, optional): Maximum number of iteration. Defaults to 100.
            tolx (float, optional): Stopping criterium for relative design change. Defaults to 1e-4.
            tolf (float, optional): Stopping criterium for relative objective change. Defaults to 1e-4.
        """
        nom = type(self).__name__
        xval = self.x
        gcur = 0.0
        while self.iter < maxiter:
            # Calculate new design
            xnew, g, dg = self.step(x=xval)

            # Check function change convergence criterion
            gprev, gcur = gcur, g
            rel_df = np.linalg.norm(gcur - gprev) / np.linalg.norm(gcur)
            if rel_df < tolf:
                if self.verbosity >= 1:
                    print(f"{nom} converged: Relative function change |Δf|/|f| ({rel_df}) below tolerance ({tolf})")
                break

            if self.verbosity >= 3:
                # Display info on variables (and sensitivities)
                self.print_variable_info(dg=(dg if  self.verbosity >= 4 else None))

            if self.verbosity >= 2:
                # Display iteration status message
                self.print_iteration_info(g, 
                                          report_feasibility=(self.verbosity >= 3), 
                                          xold=(xval if self.verbosity >= 3 else None),
                                          xnew=(xnew if self.verbosity >= 3 else None))

            # Stopping criteria on step size
            rel_stepsize = np.linalg.norm((xval - xnew) / self.dx) / np.linalg.norm(xval / self.dx)
            if rel_stepsize < tolx:
                if self.verbosity >= 1:
                    print(f"{nom} converged: Relative stepsize |Δx|/|x| ({rel_stepsize}) below tolerance ({tolx})")
                break

            xval = xnew
            self.iter += 1
            if self.verbosity >= 3:
                print("-"*50)
    
    @abstractmethod
    def step(self, 
             x: np.ndarray = None, 
             g: np.ndarray = None, 
             dg: np.ndarray = None) -> (np.ndarray, np.ndarray, np.ndarray):
        """ Performs a single optimization step

        Args:
            x (np.ndarray, optional): The design vector to evaluate. When not provided, use information set in state
            g (np.ndarray, optional): The function values to use. When not provided, will be calculated on demand
            dg (np.ndarray, optional): The sensitivity values to use. When not provided, will be calculated on demand

        Returns:
            xnew: New design vector
            g: (Evaluated) response values
            dg: (Evaluated) design sensitivities
        """
        raise NotImplementedError()


class OC(Optimizer):
    def __init__(self,
        variables: SignalsT,
        response: Signal,
        function: Network,
        slice_network=False,
        move=0.1,
        xmin=0.0,
        xmax=1.0,
        verbosity: int = 2,
        l1init: float = 0.0,
        l2init: float = 100000.0,
        l1l2tol: float = 1e-4,
        maxvol: float = None,
    ):
        """Optimality criteria optimization algorithm

        Args:
            variables: One or more variable Signals defining the design variables
            response: Response signal to be minimized (objective)
            function: The Network defining the optimization problem
        
        Keyword Args:
            slice_network (bool): If True, only the modules connecting variable and response signals are evaluated. 
              Defaults to False.
            move: Move limit on relative variable change per iteration (can be passed as scalar: same value for 
              all variables, vector: each variable has a unique value, list of scalars/vectors: for each variable 
              signal). Defaults to 0.1.
            xmin: Minimum design variable (can be passed as scalar: same value for all variables, 
              vector: each variable has a unique value, list of scalars/vectors: for each variable signal). 
              Defaults to 0.0.
            xmax: Maximum design variable (can be passed as scalar: same value for all variables, 
              vector: each variable has a unique value, list of scalars/vectors: for each variable signal). 
              Defaults to 1.0.
            verbosity (int): Level of information to print. Defaults to 2.
              0 - No prints
              1 - Only convergence message
              2 - Convergence and iteration info (default)
              3 - Additional info on variables and GCMMA inner iteration info (when applicable)
              4 - Additional info on sensitivity information
            l1init (int): Internal OC parameter. Defaults to 0.
            l2init (int): Internal OC parameter. Defaults to 100000.
            l1l2tol (float): Internal OC parameter. Defaults to 1e-4.
            maxvol (float): Volume fraction. Defaults to None.
        """
        super().__init__(variables, response, function, slice_network=slice_network, 
                         xmin=xmin, xmax=xmax, verbosity=verbosity)

        # Move limit
        if np.asarray(move).size > 1:
            self.move = self._parse_bound(move, which='move')
        else:
            self.move = move

        # Other parameters
        self.dx = self.xmax - self.xmin

        # OC parameters
        self.l1init = l1init
        self.l2init = l2init
        self.l1l2tol = l1l2tol
        self.maxvol = maxvol
    
    def step(self, 
             x: np.ndarray = None, 
             g: np.ndarray = None, 
             dg: np.ndarray = None) -> (np.ndarray, np.ndarray, np.ndarray):
        
        if x is None:
            x = self.x  # Gather the states
        else:
            self.x = x  # Set the new states
        
        if self.maxvol is None:
            self.maxvol = np.sum(x) / x.size

        if g is None:
            g = self.calculate_g()

        if dg is None:
            dg = self.calculate_dg()

        # Clip positive sensitivities
        maxdg = dg.max()
        if maxdg > 1e-15:
            warnings.warn(f"OC only works for negative sensitivities: max(dgdx) = {maxdg}. Clipping positive values.")
        dg = np.minimum(dg, 0)

        # Calculate bounds
        lb = np.maximum(self.xmin, x - self.move)
        ub = np.minimum(self.xmax, x + self.move)

        # Do OC update
        xnew = x.copy()
        l1, l2 = self.l1init, self.l2init
        while l2 - l1 > self.l1l2tol:
            lmid = 0.5 * (l1 + l2)
            xnew[:] = np.clip(x * np.sqrt(-dg / lmid), lb, ub)
            l1, l2 = (lmid, l2) if np.sum(xnew) - self.maxvol * x.size > 0 else (l1, lmid)

        return xnew, g, dg


class SLP(Optimizer):
    def __init__(self, 
                 variables: SignalsT,
                 responses: SignalsT,
                 function: Network,
                 slice_network: bool =False,
                 move=0.1,
                 xmin=0.0,
                 xmax=1.0,
                 verbosity: int = 2,
                 adaptive_movelimit: bool = True,
                 **kwargs):
        """SLP optimization algorithm
        Warning: This optimizer is experimental and is not very robust (yet)

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
          adaptive_movelimit (bool): Move limit is adapted based on variable oscillation behavior. Defaults to True.
          asyincr (float): Increase of adaptive movelimit when no oscillation is present. Default is 1.2
          asydecr (float): Decrease in adaptive movelimit when variable is oscillating. Default is 0.7
          asyinit (float): Initial adaptive movelimit value. Default is 1.0
          asybound (float): Lower bound on adaptive movelimit. Default is 1e-2
        """
        super().__init__(variables, responses, function, slice_network=slice_network, xmin=xmin, xmax=xmax)

        # For adaptive movelimit
        self.adaptive_movelimit = adaptive_movelimit
        self.asyincr = kwargs.get('asyincr', 1.2)
        self.asydecr = kwargs.get('asydecr', 0.7)
        self.asyinit = kwargs.get('asyinit', 1.0)
        self.asybound = kwargs.get('asybound', 1e-2)

        # Move limit
        if np.asarray(move).size > 1:
            self.move = self._parse_bound(move, which='move')
        else:
            self.move = move

        # Other variables
        self.xold1, self.xold2 = None, None
        self.dx = self.xmax - self.xmin
        if self.adaptive_movelimit:
            self.offset = self.asyinit * np.ones(self.n)
        else:
            self.offset = 1.0
    
    def step(self, 
             x: np.ndarray = None, 
             g: np.ndarray = None, 
             dg: np.ndarray = None) -> (np.ndarray, np.ndarray, np.ndarray):
        if x is None:
            x = self.x  # Gather the states
        else:
            self.x = x  # Set the new states

        if g is None:
            g = self.calculate_g()

        if dg is None:
            dg = self.calculate_dg()

        # Update ASYMPTOTES
        # Calculation of the asymptotes low and upp :
        # For iter = 1,2 the asymptotes are fixed depending on asyinit
        if self.adaptive_movelimit and self.xold1 is not None and self.xold2 is not None:
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
            self.offset = np.clip(self.offset, self.asybound, 1.0)  # Max offset is 1

        max_dx = self.offset * self.move * self.dx

        lb = np.maximum(self.xmin, x - max_dx)
        ub = np.minimum(self.xmax, x + max_dx)
        Ac = np.vstack(dg[1:, :]) if dg.shape[0] > 1 else None
        bc = Ac @ x - np.hstack(g[1:]) if dg.shape[0] > 1 else None
        
        res = linprog(dg[0, :], Ac, bc, bounds=np.vstack([lb, ub]).T, options={"disp": False})
        if not res.success:
            print(res)
            return x, g, dg
        xnew = res.x

        # Update design vector
        self.xold2, self.xold1 = self.xold1, x

        return xnew, g, dg