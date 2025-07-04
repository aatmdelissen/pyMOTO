import sys
import warnings
import inspect
import time
import copy
from typing import Union, List, Any
from abc import ABC, abstractmethod
from collections.abc import Callable
from .utils import _parse_to_list, _concatenate_to_array, _split_from_array


# Local helper functions
def err_fmt(*args):
    """ Format error strings for locating Modules and Signals"""
    err_str = ""
    for a in args:
        err_str += f"\n\t| {a}"
    return err_str


def colored(r, g, b, text):
    """ Colored console text """
    return "\033[38;2;{};{};{}m{} \033[0;0m".format(r, g, b, text)


def stderr_warning(text):
    """ Issue a (colored) warning """
    filename, line, func = get_init_loc()
    warnings.warn_explicit(colored(128, 128, 0, "\nWarning: " + text), RuntimeWarning, filename, line)


def get_init_loc():
    """ Get the location (outside of this file) where the 'current' function is called """
    stk = inspect.stack()
    frame = None
    for fr in stk:
        if fr[1] != __file__:
            frame = fr
            break
    if frame is None:
        return "N/A", "N/A", "N/A"
    _, filename, line, func, _, _ = frame
    return filename, line, func


def get_init_str():
    filename, line, func = get_init_loc()
    return f"File \"{filename}\", line {line}, in {func}"


def fmt_slice(sl):
    """ Formats slices as string
    :param sl: Generic slice or tuple of slices
    :return: Slice(s) formatted as string
    """
    if isinstance(sl, tuple):
        return ", ".join([fmt_slice(sli) for sli in sl])
    elif isinstance(sl, slice):
        return f"{'' if sl.start is None else sl.start}:" \
               f"{'' if sl.stop is None else sl.stop}:" \
               f"{'' if sl.step is None else sl.step}"
    elif hasattr(sl, 'size') and sl.size > 4:
        ndim = sl.ndim if hasattr(sl, 'ndim') else 1
        return "["*ndim + "..." + "]"*ndim
    else:
        return str(sl)


class Signal:
    """
    Manages the state data, sensitivities, and connects module in- and outputs

    Initialize using Signal()
    Optional arguments: tag (string)
    Optional keyword arguments: tag=(string)

    >> Signal('x1')

    >> Signal(tag='x2')

    """
    def __init__(self, tag: str = "", state: Any = None, sensitivity: Any = None, min: Any = None, max: Any = None):
        """
        Keyword Args:
            tag: The name of the signal
            state: The initialized state
            sensitivity: The initialized sensitivity
            min: Minimum allowed value
            max: Maximum allowed value
        """
        self.tag = tag
        self.state = state
        self.sensitivity = sensitivity
        self.min = min
        self.max = max
        self.keep_alloc = sensitivity is not None

        # Save error string to location where it is initialized
        self._init_loc = get_init_str()

    def _err_str(self):
        return err_fmt(f"Signal \'{self.tag}\', initialized in {self._init_loc}")

    def add_sensitivity(self, ds: Any):
        """ Add a new term to internal sensitivity """
        try:
            if ds is None:
                return
            if self.sensitivity is None:
                self.sensitivity = copy.deepcopy(ds)
            elif hasattr(self.sensitivity, "add_sensitivity"):
                # Allow user to implement a custom add_sensitivity function instead of __iadd__
                self.sensitivity.add_sensitivity(ds)
            else:
                self.sensitivity += ds
            return self
        except TypeError:
            if isinstance(ds, type(self.sensitivity)):
                raise TypeError(f"Cannot add to the sensitivity with type '{type(self.sensitivity).__name__}'"+self._err_str())
            else:
                raise TypeError(f"Adding wrong type '{type(ds).__name__}' to the sensitivity '{type(self.sensitivity).__name__}'"+self._err_str())
        except ValueError:
            sens_shape = self.sensitivity.shape if hasattr(self.sensitivity, 'shape') else ()
            ds_shape = ds.shape if hasattr(ds, 'shape') else ()
            raise ValueError(f"Cannot add argument of shape {ds_shape} to the sensitivity of shape {sens_shape}"+self._err_str()) from None

    def reset(self, keep_alloc: bool = None):
        """ Reset the sensitivities to zero or None
        This must be called to clear internal memory of subsequent sensitivity calculations.

        Args:
            keep_alloc: Keep the sensitivity allocation intact?

        Returns:
            self
        """
        if self.sensitivity is None:
            return self
        if keep_alloc is None:
            keep_alloc = self.keep_alloc
        if keep_alloc:
            try:
                try:
                    self.sensitivity[...] = 0
                except TypeError:
                    self.sensitivity *= 0
            except TypeError:
                stderr_warning(f"reset() - Cannot keep allocation because the operands *= or [] are not defined for sensitivity type \'{type(self.sensitivity).__name__}\'" + self._err_str())
                self.sensitivity = None
        else:
            self.sensitivity = None
        return self

    def __getitem__(self, item):
        """ Obtain a sliced signal, for using its partial contents.

        Args:
            item: Slice indices

        Returns:
            Sliced signal (SignalSlice)
        """
        return SignalSlice(self, item)

    def __str__(self):
        state_msg = f"state {self.state}" if self.state is not None else "empty state"
        state_msg = state_msg.split('\n')
        if len(state_msg) > 1:
            state_msg = state_msg[0] + ' ... ' + state_msg[-1]
        else:
            state_msg = state_msg[0]
        return f"Signal \"{self.tag}\" with {state_msg}"

    def __repr__(self):
        state_msg = f"state {self.state}" if self.state is not None else "empty state"
        state_msg = state_msg.split('\n')
        if len(state_msg) > 1:
            state_msg = state_msg[0] + ' ... ' + state_msg[-1]
        else:
            state_msg = state_msg[0]
        sens_msg = 'empty sensitivity' if self.sensitivity is None else 'non-empty sensitivity'
        return f"Signal \"{self.tag}\" with {state_msg} and {sens_msg} at {hex(id(self))}"


class SignalSlice(Signal):
    """ Slice operator for a Signal
    The sliced values are referenced to their original source Signal, such that they can be used and updated in modules.
    This means that updating the values in this SignalSlice changes the data in its source Signal.
    """
    def __init__(self, base, sl, tag=None):
        self.base = base
        self.slice = sl
        self.keep_alloc = False  # Allocation must be False because sensitivity cannot be assigned with [] operator

        # for s in slice:
        if tag is None:
            self.tag = f"{self.base.tag}[{fmt_slice(self.slice)}]"
        else:
            self.tag = tag

        # Save error string to location where it is initialized
        self._init_loc = get_init_str()

    @property
    def state(self):
        try:
            return None if self.base.state is None else self.base.state[self.slice]
        except Exception as e:
            # Possibilities: Unslicable object (TypeError) or Wrong dimensions or out of range (IndexError)
            raise type(e)(str(e) + "\n\t| Above error was raised in SignalSlice.state (getter). Signal details:" +
                      self._err_str()).with_traceback(sys.exc_info()[2])

    @state.setter
    def state(self, new_state):
        try:
            self.base.state[self.slice] = new_state
        except Exception as e:
            # Possibilities: Unslicable object (TypeError) or Wrong dimensions or out of range (IndexError)
            raise type(e)(str(e) + "\n\t| Above error was raised in SignalSlice.state (setter). Signal details:" +
                          self._err_str()).with_traceback(sys.exc_info()[2])

    @property
    def sensitivity(self):
        try:
            return None if self.base.sensitivity is None else self.base.sensitivity[self.slice]
        except Exception as e:
            # Possibilities: Unslicable object (TypeError) or Wrong dimensions or out of range (IndexError)
            raise type(e)(str(e) + "\n\t| Above error was raised in SignalSlice.sensitivity (getter). Signal details:" +
                          self._err_str()).with_traceback(sys.exc_info()[2])

    @sensitivity.setter
    def sensitivity(self, new_sens):
        try:
            if self.base.sensitivity is None:
                # Initialize sensitivity of base-signal
                if new_sens is None:
                    return  # Sensitivity doesn't need to be initialized when it is set to None
                try:
                    self.base.sensitivity = self.base.state * 0  # Make a new copy with 0 values
                except TypeError:
                    if self.base.state is None:
                        raise TypeError("Could not initialize sensitivity because state is not set" + self._err_str())
                    else:
                        raise TypeError(f"Could not initialize sensitivity for type \'{type(self.base.state).__name__}\'")

            if new_sens is None:
                new_sens = 0  # reset() uses this

            self.base.sensitivity[self.slice] = new_sens
        except Exception as e:
            # Possibilities: Unslicable object (TypeError) or Wrong dimensions or out of range (IndexError)
            raise type(e)(str(e) + "\n\t| Above error was raised in SignalSlice.state (setter). Signal details:" +
                          self._err_str()).with_traceback(sys.exc_info()[2])

    def add_sensitivity(self, ds: Any):
        """ Add a new term to internal sensitivity """
        try:
            if ds is None:
                return
            if self.base.sensitivity is None:
                self.base.sensitivity = self.base.state * 0
                # self.sensitivity = copy.deepcopy(ds)

            if hasattr(self.sensitivity, "add_sensitivity"):
                # Allow user to implement a custom add_sensitivity function instead of __iadd__
                self.sensitivity.add_sensitivity(ds)
            else:
                self.sensitivity += ds
            return self
        except TypeError:
            if isinstance(ds, type(self.sensitivity)):
                raise TypeError(
                    f"Cannot add to the sensitivity with type '{type(self.sensitivity).__name__}'" + self._err_str())
            else:
                raise TypeError(
                    f"Adding wrong type '{type(ds).__name__}' to the sensitivity '{type(self.sensitivity).__name__}'" + self._err_str())
        except ValueError:
            sens_shape = self.sensitivity.shape if hasattr(self.sensitivity, 'shape') else ()
            ds_shape = ds.shape if hasattr(ds, 'shape') else ()
            raise ValueError(
                f"Cannot add argument of shape {ds_shape} to the sensitivity of shape {sens_shape}" + self._err_str()) from None

    def reset(self, keep_alloc: bool = None):
        """ Reset the sensitivities to zero or None
        This must be called to clear internal memory of subsequent sensitivity calculations.
        :param keep_alloc: Keep the sensitivity allocation intact?
        :return: self
        """
        if self.sensitivity is not None:
            self.sensitivity = None
        return self


def make_signals(*args):
    """ Batch-initialize a number of Signals
    :param args: Tags for a number of Signals
    :return: Dictionary of Signals, with key index equal to the signal tag
    """
    ret = dict()
    for a in args:
        ret[a] = Signal(a)
    return ret


def _is_valid_signal(sig: Any):
    """ Checks if the argument is a valid Signal object
    :param sig: The object to check
    :return: True if it is a valid Signal
    """
    if isinstance(sig, Signal):
        return True
    if all([hasattr(sig, f) for f in ["state", "sensitivity", "add_sensitivity", "reset"]]):
        return True
    return False


def _is_valid_module(mod: Any):
    """ Checks if the argument is a valid Module object
    :param mod: The object to check
    :return: True if it is a valid Module
    """
    if issubclass(type(mod), Module):
        return True
    if hasattr(mod, "response") and hasattr(mod, "sensitivity") and hasattr(mod, "reset"):
        return True
    return False


# Type definition for bound method
class BoundMethod:
    __self__: object


BoundMethodT = Union[Callable, BoundMethod]


def _check_function_signature(fn: BoundMethodT, signals: list = None) -> (int, int):
    """ Checks the function signature against given signal list

    - Only positional-only or positional-or-keyword arguments are allowed
      (https://peps.python.org/pep-0362/#parameter-object)
    - If `signals` is provided, the number of them is compared with the allowed min/max number of arguments

    Args:
        fn: response_ or sensitivity_ function of a Module
        signals (optional): The signals involved, which are checked with the function signature

    Returns:
        (min_args, max_args): Minimum and maximum number of accepted arguments to the function;
          `None` denotes an unlimited number of maximum arguments (caused by `*args`)
    """
    min_args, max_args = 0, 0
    callstr = f"{type(fn.__self__).__name__}.{fn.__name__}{inspect.signature(fn)}"
    # Loop over all the parameters defined for the function
    for s, p in inspect.signature(fn).parameters.items():
        if p.kind == inspect.Parameter.POSITIONAL_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if p.default == inspect.Parameter.empty:
                # No default argument is provided, a value MUST be provided
                min_args += 1
            if max_args is not None:
                max_args += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            # If *args is passed, the maximum number of arguments is unknown. Does not affect minimum number
            max_args = None
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            raise SyntaxError(f"{callstr} may not contain keyword-only arguments \"{p}\"")
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            raise SyntaxError(f"{callstr} may not contain dict of keyword-only arguments \"{p}\"")

    # Check with signal list
    if signals is not None:
        if len(signals) < min_args:
            raise TypeError(f"Not enough arguments ({len(signals)}) for {callstr}; expected {min_args}")
        if max_args is not None and len(signals) > max_args:
            raise TypeError(f"Too many arguments ({len(signals)}) for {callstr}; expected {max_args}")
    return min_args, max_args


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
    >> Module(sig_in=[inputs], sig_out=[outputs])
    """

    def _err_str(self, module_signature: bool = True, init: bool = True, fn=None):
        str_list = []

        if module_signature:
            inp_str = "Inputs: " + ", ".join([s.tag if hasattr(s, 'tag') else 'N/A' for s in self.sig_in]) if len(self.sig_in) > 0 else "No inputs"
            out_str = "Outputs: " + ", ".join([s.tag if hasattr(s, 'tag') else 'N/A' for s in self.sig_out]) if len(self.sig_out) > 0 else "No outputs"
            str_list.append(f"Module \'{type(self).__name__}\'( " + inp_str + " ) --> " + out_str)
        if init:
            str_list.append(f"Used in {self._init_loc}")
        if fn is not None:
            name = f"{fn.__self__.__class__.__name__}.{fn.__name__}{inspect.signature(fn)}"
            lineno = inspect.getsourcelines(fn)[1]
            filename = inspect.getfile(fn)
            str_list.append(f"Implementation in File \"{filename}\", line {lineno}, in {name}")
        return err_fmt(*str_list)

    # flake8: noqa: C901
    def __init__(self, sig_in: Union[Signal, List[Signal]] = None, sig_out: Union[Signal, List[Signal]] = None,
                 *args, **kwargs):
        self._init_loc = get_init_str()

        self.sig_in = _parse_to_list(sig_in)
        self.sig_out = _parse_to_list(sig_out)
        for i, s in enumerate(self.sig_in):
            if not _is_valid_signal(s):
                tag = f" (\'{s.tag}\')" if hasattr(s, 'tag') else ''
                raise TypeError(f"Input {i}{tag} is not a valid signal, type=\'{type(s).__name__}\'.")

        for i, s in enumerate(self.sig_out):
            if not _is_valid_signal(s):
                tag = f" (\'{s.tag}\')" if hasattr(s, 'tag') else ''
                raise TypeError(f"Output {i}{tag} is not a valid signal, type=\'{type(s).__name__}\'.")

        # Call preparation of submodule with remaining arguments
        self._prepare(*args, **kwargs)

        try:
            # Check if the signals match _response() signature
            _check_function_signature(self._response, self.sig_in)
        except Exception as e:
            raise type(e)(str(e) + "\n\t| Module details:" +
                          self._err_str(fn=self._response)).with_traceback(sys.exc_info()[2])

        try:
            # If no output signals are given, but are required, try to initialize them here
            req_args = 0
            for s, p in inspect.signature(self._sensitivity).parameters.items():
                if p.kind == inspect.Parameter.POSITIONAL_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    if req_args >= 0:
                        req_args += 1
                elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                    req_args = -1
            if len(self.sig_out) == 0 and req_args >= 0 and req_args != len(self.sig_out):
                # Initialize a number of output signals with default names
                self.sig_out = [Signal(f"{type(self).__name__}_output{i}") for i in range(req_args)]

            # Check if signals match _sensitivity() signature
            _check_function_signature(self._sensitivity, self.sig_out)
        except Exception as e:
            raise type(e)(str(e) + "\n\t| Module details:" +
                          self._err_str(fn=self._sensitivity)).with_traceback(sys.exc_info()[2])

    def response(self):
        """ Calculate the response from sig_in and output this to sig_out """
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
            return self
        except Exception as e:
            # https://stackoverflow.com/questions/6062576/adding-information-to-an-exception
            raise type(e)(str(e) + "\n\t| Above error was raised when calling response(). Module details:" +
                          self._err_str(fn=self._response)).with_traceback(sys.exc_info()[2])

    def __call__(self, *args):
        if len(args) == 0 and len(self.sig_in) > 0:
            inp = [s.state for s in self.sig_in]
            return self._response(*inp)
        else:
            return self._response(*args)

    def sensitivity(self):
        """  Calculate sensitivities using backpropagation

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

            return self
        except Exception as e:
            raise type(e)(str(e) + "\n\t| Above error was raised when calling sensitivity(). Module details:" +
                          self._err_str(fn=self._sensitivity)).with_traceback(sys.exc_info()[2])

    def reset(self):
        """ Reset the state of the sensitivities (they are set to zero or to None) """
        try:
            [s.reset() for s in self.sig_out]
            [s.reset() for s in self.sig_in]
            self._reset()
            return self
        except Exception as e:
            raise type(e)(str(e) + "\n\t| Above error was raised when calling reset(). Module details:" +
                          self._err_str(fn=self._response)).with_traceback(sys.exc_info()[2])

    # METHODS TO BE DEFINED BY USER
    def _prepare(self, *args, **kwargs):
        pass

    @abstractmethod
    def _response(self, *args):
        raise NotImplementedError("No response behavior defined")

    def _sensitivity(self, *args):
        if len(self.sig_out) > 0 and len(self.sig_in) > 0:
            stderr_warning(f"Sensitivity routine is used, but not defined, in {type(self).__name__}")
        return [None for _ in self.sig_in]

    def _reset(self):
        pass


class Network(Module):
    """ Binds multiple Modules together as one Module

    Initialize a network with a number of modules that should be executed consecutively
    >> Network(module1, module2, ...)

    >> Network([module1, module2, ...])

    >> Network((module1, module2, ...))

    Modules can also be constructed using a dictionary based on strings
    >> Network([ {type="module1", sig_in=[sig1, sig2], sig_out=[sig3]},
                 {type="module2", sig_in=[sig3], sig_out=[sig4]} ])

    Appending modules to a network will output the signals automatically
    >> fn = Network()
    >> s_out = fn.append(module1)

    Args:
        print_timing: Print timing of each module inside this Network
    """
    def __init__(self, *args, print_timing=False):
        super().__init__()
        self._init_loc = get_init_str()
        self.mods = []  # Empty module list
        self.append(*args)  # Append to module list
        self.print_timing = print_timing

    def timefn(self, fn, name=None):
        start_t = time.time()
        fn()
        duration = time.time() - start_t
        if name is None:
            name = f"{fn}"
        if isinstance(self.print_timing, bool):
            tmin = 0.0
        else:
            tmin = self.print_timing
        if duration > tmin:
            print(f"{name} took {time.time() - start_t} s")

    def response(self):
        if self.print_timing is not False:
            start_t = time.time()
            [self.timefn(m.response, name=f"-- Response of \"{type(m).__name__}\"") for m in self.mods]
            duration = time.time() - start_t
            if isinstance(self.print_timing, bool):
                tmin = 0.0
            else:
                tmin = self.print_timing
            if duration > tmin:
                print(f"-- TOTAL Response took {time.time() - start_t} s")
        else:
            [m.response() for m in self.mods]

    def sensitivity(self):
        if self.print_timing is not False:
            start_t = time.time()
            [self.timefn(m.sensitivity, name=f"-- Sensitivity of \"{type(m).__name__}\"") for m in reversed(self.mods)]
            duration = time.time() - start_t
            if isinstance(self.print_timing, bool):
                tmin = 0.0
            else:
                tmin = self.print_timing
            if duration > tmin:
                print(f"-- TOTAL Sensitivity took {time.time() - start_t} s")
        else:
            [m.sensitivity() for m in reversed(self.mods)]

    def reset(self):
        [m.reset() for m in reversed(self.mods)]

    def _response(self, *args):
        pass  # Unused

    def __copy__(self):
        return Network(*self.mods)

    def copy(self):
        return self.__copy__()

    def __len__(self):
        return len(self.mods)

    def __getitem__(self, i):
        return self.mods[i]

    def __iter__(self):
        return iter(self.mods)

    def __call__(self, *args):
        return self.append(*args)

    def append(self, *newmods):
        modlist = _parse_to_list(*newmods)
        if len(modlist) == 0:
            return

        # Check if the blocks are initialized, else create them
        for i, m in enumerate(modlist):
            if isinstance(m, dict):
                exclude_keys = ['type']
                b_ex = {k: m[k] for k in set(list(m.keys())) - set(exclude_keys)}
                modlist[i] = Module.create(m['type'], **b_ex)

        # Check validity of modules
        for i, m in enumerate(modlist):
            if not _is_valid_module(m):
                raise TypeError(f"Argument #{i} is not a valid module, type=\'{type(m).__name__}\'.")

        # Obtain the internal blocks
        self.mods.extend(modlist)

        # Gather all the input and output signals of the internal blocks
        all_in = set()
        all_out = set()
        [all_in.update(m.sig_in) for m in self.mods]
        [all_out.update(m.sig_out) for m in self.mods]
        in_unique = all_in - all_out

        self.sig_in = _parse_to_list(in_unique)
        self.sig_out = _parse_to_list(all_out)

        return modlist[-1].sig_out[0] if len(modlist[-1].sig_out) == 1 else modlist[-1].sig_out
