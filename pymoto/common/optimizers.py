from abc import ABC, abstractmethod
import numpy as np
from ..utils import _parse_to_list, _concatenate_to_array
from ..core_objects import Network


class Optimizer(ABC):
    """General abstract optimizer object"""
    def __init__(self, variables, responses, function=None, slice_network=False, xmin=None, xmax=None):
        self.variables = _parse_to_list(variables)
        self.responses = _parse_to_list(responses)
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
    
    @abstractmethod
    def optimize(self, maxiter: int = 100, tolx: float = 1e-4, tolf: float = 1e-4):
        """ Perform an optimization 

        Args:
            maxiter (int, optional): Maximum number of iteration. Defaults to 100.
            tolx (float, optional): Stopping criterium for relative design change. Defaults to 1e-4.
            tolf (float, optional): Stopping criterium for relative objective change. Defaults to 1e-4.
        """
        raise NotImplementedError()
    
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