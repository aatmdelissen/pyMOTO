from typing import Union, List, Any
import warnings
import sys
import inspect
from .utils import _parse_to_list
from abc import ABC, abstractmethod


class Signal:
    """
    Saves the state data, connects input and outputs of blocks and manages sensitivities

    Initialize using Signal()
    Optional arguments: tag (string)
    Optional keyword arguments: tag=(string)

    >> Signal('x1')

    >> Signal(tag='x2')

    """
    def __init__(self, tag: str = "", state: Any = None, sensitivity: Any = None):
        """

        :param tag: The name of the signal (string)
        :param state: The initialized state (optional)
        :param sensitivity: The initialized sensitivity (optional)
        """
        self.tag = tag
        self.state = state
        self.sensitivity = sensitivity
        self.keep_alloc = sensitivity is not None

    def add_sensitivity(self, ds: Any):
        try:
            if ds is None:
                return
            if self.sensitivity is None:
                self.sensitivity = ds
            else:
                self.sensitivity += ds
        except Exception as e:
            raise type(e)(f"{e}, in signal {self.tag}").with_traceback(sys.exc_info()[2]) from None

    def reset(self, keep_alloc: bool = False):
        if self.keep_alloc and keep_alloc and self.sensitivity is not None:
            self.sensitivity *= 0
        else:
            self.sensitivity = None


def make_signals(*args):
    """ Batch-initialize a number of Signals

    :param args: Tags for a number of Signals
    :return: List of Signals
    """
    return [Signal(s) for s in args]


def _check_valid_signal(sig: Any):
    """ Checks if the argument is a valid Signal object

    :param sig: The object to check
    :return: True if it is a valid Signal
    """
    if isinstance(sig, Signal):
        return True
    if hasattr(sig, "state") and hasattr(sig, "sensitivity"):
        return True
    if hasattr(sig, "add_sensitivity"):
        return True
    raise TypeError("Entry {} is not a Signal".format(sig))


def _check_valid_module(mod: Any):
    """ Checks if the argument is a valid Module object

    :param mod: The object to check
    :return: True if it is a valid Module
    """
    if issubclass(type(mod), Module):
        return True
    if hasattr(mod, "response") and hasattr(mod, "sensitivity") and hasattr(mod, "reset"):
        return True
    raise TypeError("Entry {} is not a Module".format(mod))


def _check_function_signature(fn, signals):
    """ Checks the function signature against given signal list

    :param fn: response_ or sensitivity_ function of a Module
    :param signals: The signals involved
    """
    min_args, max_args = 0, 0
    callstr = f"{type(fn.__self__).__name__}.{fn.__name__}{inspect.signature(fn)}"
    for s, p in inspect.signature(fn).parameters.items():
        if p.kind==inspect.Parameter.POSITIONAL_ONLY or p.kind==inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if p.default!=inspect.Parameter.empty:
                raise SyntaxError(f"{callstr} must not have default values \"{p}\"")
            min_args += 1
            if max_args>=0:
                max_args += 1
        elif p.kind==inspect.Parameter.VAR_POSITIONAL:
            max_args = -1
        elif p.kind==inspect.Parameter.KEYWORD_ONLY:
            raise SyntaxError(f"{callstr} may not contain keyword arguments \"{p}\"")
        elif p.kind==inspect.Parameter.VAR_KEYWORD:
            raise SyntaxError(f"{callstr} may not contain \"{p}\"")
    if len(signals) < min_args:
        raise TypeError(f"Not enough arguments ({len(signals)}) for {callstr}")
    if max_args >=0 and len(signals) > max_args:
        raise TypeError(f"Too many arguments ({len(signals)}) for {callstr}")


class RegisteredClass(object):
    """
    Abstract base class that can keep track of its subclasses and can instantiate them as well, based on their name.
    """

    @classmethod
    def create(cls, sub_type: str, *args, **kwargs):
        """
        Factory method to create subclasses. Call with the name of the subclass to instantiate a new object of
        requested type.
        :param sub_type: String identifying the name of the subclass (case insensitive)
        :param args: Passed to the subclass constructor
        :param kwargs: Passed to the subclass constructor
        :return: New subclass object
        """
        id_req = sub_type.lower()
        subs = cls.all_subclasses()
        if id_req not in subs.keys():
            raise ValueError("Subclass type not defined: {}".format(sub_type))
        return subs[id_req](*args, **kwargs)

    @classmethod
    def all_subclasses(cls):
        """
        Looks for subclasses of this class, used in creation
        :return: List of (unique) subclasses
        """

        # Recursive search for subclasses
        def get_subs(cl):
            all_subclasses = []

            for subclass in cl.__subclasses__():
                all_subclasses.append(subclass)
                all_subclasses.extend(get_subs(subclass))

            return all_subclasses

        subs = get_subs(cls)

        # Check for duplicates
        seen = {}
        subs_out = {}
        duplicates = []
        duplicate_obj = []
        for sc in subs:
            scn = sc.__name__.lower()
            if scn not in seen:
                seen[scn] = 1
                subs_out[scn] = sc
            else:
                if seen[scn] > 0:
                    duplicates.append(scn)
                    duplicate_obj.append(sc)
                seen[scn] += 1

        # Emit warning if duplicates are found
        for d, do in zip(duplicates, duplicate_obj):
            warnings.warn("Duplicated module '{}', currently defined as {}, duplicate definition at {}"
                          .format(d, subs_out[d], do), Warning)

        return subs_out

    @classmethod
    def print_children(cls):
        print(": ".join([cls.__name__+" subtypes", ", ".join(cls.all_subclasses().keys())]))


class Module(ABC, RegisteredClass):
    """
    Main class: Module
    Transforms input signal to output signal and output signal sensitivity to input signal sensitivity

    Initialize using Module()
    >> Module(input, output)

    Multiple inputs and outputs:
    >> Module([input1, input2], [output1, output2])

    No tag:
    >> Module([input1, input2], [output1, output2])

    No outputs:
    >> Module([inputs])

    Using keywords:
    >> Module(sig_in=[inputs], sig_out=[outputs]
    """

    def _error_str(self, add_signal: bool=True, fn=None):
        sig_str = ""
        if add_signal:
            sig_str = f"{[s.tag for s in self.sig_in]}->{[s.tag for s in self.sig_out]}"

        if fn is None:
            name = self.__class__.__name__
            lineno = inspect.getsourcelines(self.__class__)[1]
            filename = inspect.getfile(self.__class__)
        else:
            name = f"{fn.__self__.__class__.__name__}.{fn.__name__}{inspect.signature(fn)}"
            lineno = inspect.getsourcelines(fn)[1]
            filename = inspect.getfile(fn)
        return f"\n\tModule File \"{filename}\", line {lineno}, in {name} {sig_str}"

    def __init__(self, sig_in: Union[Signal, List[Signal]] = None, sig_out: Union[Signal, List[Signal]] = None,
                 *args, **kwargs):
        try:
            # Parse input and output signals
            self.sig_in = _parse_to_list(sig_in)
            self.sig_out = _parse_to_list(sig_out)

            # Check if the signals are valid
            [_check_valid_signal(s) for s in self.sig_in]
            [_check_valid_signal(s) for s in self.sig_out]
        except Exception as e:
            raise type(e)(str(e) + self._error_str(add_signal=True)).with_traceback(sys.exc_info()[2]) from None

        try:
            # Call preparation of submodule with remaining arguments
            self._prepare(*args, **kwargs)

            # Save error string to location where it is initialized
            _, filename, line, func, _, _ = inspect.stack()[1]
            self._init_err_str = f"\n\tInitialized in File \"{filename}\", line {line}, in {func}"
        except Exception as e:
            raise type(e)(str(e) + self._error_str(add_signal=False)).with_traceback(sys.exc_info()[2]) from None

        try:
            # Check if the signals match _response() signature
            _check_function_signature(self._response, self.sig_in)
        except Exception as e:
            raise type(e)(str(e) + self._error_str(add_signal=False, fn=self._response)).with_traceback(sys.exc_info()[2]) from None

        try:
            # If no output signals are given, but are required, try to initialize them here
            req_args = 0
            for s, p in inspect.signature(self._sensitivity).parameters.items():
                if p.kind==inspect.Parameter.POSITIONAL_ONLY or p.kind==inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    if req_args>=0:
                        req_args += 1
                elif p.kind==inspect.Parameter.VAR_POSITIONAL:
                    req_args = -1
            if len(self.sig_out)==0 and req_args>=0 and req_args != len(self.sig_out):
                # Initialize a number of output signals with default names
                self.sig_out = [Signal(f"{type(self).__name__}_output{i}") for i in range(req_args)]

            # Check if signals match _sensitivity() signature
            _check_function_signature(self._sensitivity, self.sig_out)
        except Exception as e:
            raise type(e)(str(e) + self._error_str(add_signal=False, fn=self._sensitivity)).with_traceback(sys.exc_info()[2]) from None



    def response(self):
        """
        Calculate the response from sig_in and output this to sig_out
        """
        try:
            inp = [s.state for s in self.sig_in]
            state_out = _parse_to_list(self._response(*inp))  # Calculate the actual response

            # Check if enough outputs are calculated
            if len(state_out) != len(self.sig_out):
                raise TypeError(f"Number of responses calculated ({len(state_out)}) is unequal to "
                                f"number of output signals ({len(self.sig_out)})")

            # Update the output signals
            for i, val in enumerate(state_out):
                self.sig_out[i].state = val

        except Exception as e:
            raise type(e)(str(e) + self._error_str(fn=self._response) + self._init_err_str).with_traceback(sys.exc_info()[2]) from None

    def sensitivity(self):
        """
        Based on the sensitivity we get from sig_out, reverse the process and output the new sensitivities to sig_in
        """
        try:
            # Get all sensitivity values of the outputs
            sens_in = [s.sensitivity for s in self.sig_out]

            if len(self.sig_out) > 0 and all([s is None for s in sens_in]):
                return  # If none of the adjoint variables is set

            # Calculate the new sensitivities of the inputs
            sens_out = _parse_to_list(self._sensitivity(*sens_in))

            # Check if enough sensitivities are calculated
            if len(sens_out) != len(self.sig_in):
                raise TypeError(f"Number of sensitivities calculated ({len(sens_out)}) is unequal to "
                                f"number of input signals ({len(self.sig_in)})")

            # Add the sensitivities to the signals
            for i, ds in enumerate(sens_out):
                self.sig_in[i].add_sensitivity(ds)

        except Exception as e:
            raise type(e)(str(e) + self._error_str(fn=self._sensitivity) + self._init_err_str).with_traceback(sys.exc_info()[2]) from None

    def reset(self):
        try:
            [s.reset() for s in self.sig_out]
            [s.reset() for s in self.sig_in]
            self._reset()
        except Exception as e:
            raise type(e)(str(e) + self._error_str(fn=self._reset)).with_traceback(sys.exc_info()[2]) from None

    # METHODS TO BE DEFINED BY USER
    def _prepare(self, *args, **kwargs):
        pass

    @abstractmethod
    def _response(self, *args):
        raise NotImplementedError("No response behavior defined")

    def _sensitivity(self, *args):
        if len(self.sig_out) > 0 and len(self.sig_in) > 0:
            warnings.warn("Sensitivity routine is used, but not defined, in {}".format(type(self).__name__), Warning)
        return [None for _ in self.sig_in]

    def _reset(self):
        pass

try:
    import jax
    import numpy as np
    class AutoMod(Module):
        """ Module that automatically differentiates the response function """
        def response(self):
            # Calculate the response and tangent operator (JAX Vector-Jacobian product)
            y, self.vjp_fn = jax.vjp(self._response, *[s.state for s in self.sig_in])
            y = _parse_to_list(y)

            # Assign all the states
            for i, s in enumerate(self.sig_out):
                s.state = y[i]

        def sensitivity(self):
            # Gather the output sensitivities
            dfdv = [s.sensitivity for s in self.sig_out]
            if np.all([df is None for df in dfdv]):
                return
            for i in range(len(dfdv)):
                if dfdv[i] is None:  # JAX does not accept None as 0
                    dfdv[i] = np.zeros_like(self.sig_out[i].state)
                elif np.iscomplexobj(dfdv[i]):
                    dfdv[i] = np.conj(dfdv[i])
            dfdv = tuple(dfdv) if len(dfdv) > 1 else dfdv[0]

            # Calculate backward sensitivity
            dfdx = _parse_to_list(self.vjp_fn(dfdv))

            # Assign the sensitivities
            for i, s in enumerate(self.sig_in):
                if np.iscomplexobj(dfdx[i]):
                    s.add_sensitivity(np.conj(dfdx[i]))
                else:
                    s.add_sensitivity(dfdx[i])
except ImportError:
    class AutoMod(Module):
        def __init__(self):
            super().__init__()
            raise ImportError("Could not create this object, as it is dependent on the Python library \"jax\"")


class Network(Module):
    """ Binds multiple Modules together as one Module

    >> Network(module1, module2, ...)

    >> Network([module1, module2, ...])

    >> Network((module1, module2, ...))

    >> Network([ {type="module1", sig_in=[sig1, sig2], sig_out=[sig3]},
                 {type="module2", sig_in=[sig3], sig_out=[sig4]} ])

    """
    def __init__(self, *args):
        try:
            # Obtain the internal blocks
            self.mods = _parse_to_list(*args)

            # Check if the blocks are initialized, else create them
            for i, b in enumerate(self.mods):
                if isinstance(b, dict):
                    exclude_keys = ['type']
                    b_ex = {k: b[k] for k in set(list(b.keys())) - set(exclude_keys)}
                    self.mods[i] = Module.create(b['type'], **b_ex)

            # Check validity of modules
            [_check_valid_module(m) for m in self.mods]

            # Gather all the input and output signals of the internal blocks
            all_in = set()
            all_out = set()
            [all_in.update(b.sig_in) for b in self.mods]
            [all_out.update(b.sig_out) for b in self.mods]
            in_unique = all_in - all_out

            # Initialize the parent module, with correct inputs and outputs
            super().__init__(list(in_unique), list(all_out))
        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def response(self):
        [b.response() for b in self.mods]

    def sensitivity(self):
        [b.sensitivity() for b in reversed(self.mods)]

    def reset(self):
        [b.reset() for b in reversed(self.mods)]

    def _response(self, *args):
        pass  # Unused

    def __getitem__(self, item):
        return self.mods[item]

    def append(self, *newmods):
        modlist = _parse_to_list(*newmods)
        # Obtain the internal blocks
        self.mods.extend(modlist)

        # Check if the blocks are initialized, else create them
        for i, b in enumerate(self.mods):
            if isinstance(b, dict):
                exclude_keys = ['type']
                b_ex = {k: b[k] for k in set(list(b.keys())) - set(exclude_keys)}
                self.mods[i] = Module.create(b['type'], **b_ex)

        # Check validity of modules
        [_check_valid_module(m) for m in self.mods]

        # Gather all the input and output signals of the internal blocks
        all_in = set()
        all_out = set()
        [all_in.update(b.sig_in) for b in self.mods]
        [all_out.update(b.sig_out) for b in self.mods]
        in_unique = all_in - all_out

        self.sig_in = _parse_to_list(in_unique)
        [_check_valid_signal(s) for s in self.sig_in]
        self.sig_out = _parse_to_list(all_out)
        [_check_valid_signal(s) for s in self.sig_out]

        return modlist[-1].sig_out[0] if len(modlist[-1].sig_out)==1 else modlist[-1].sig_out
