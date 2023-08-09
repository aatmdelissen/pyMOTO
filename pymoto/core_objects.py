from typing import Union, List, Any
import warnings
import inspect
import time
from .utils import _parse_to_list, _concatenate_to_array, _split_from_array
from abc import ABC, abstractmethod


# Local helper functions
def err_fmt(*args):
    """ Format error strings for locating Modules and Signals"""
    err_str = ""
    for a in args:
        err_str += f"\n\t[ {a} ]"
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

        # Save error string to location where it is initialized
        self._init_loc = get_init_str()

    def _err_str(self):
        return err_fmt(f"Signal \'{self.tag}\', initialized in {self._init_loc}")

    def add_sensitivity(self, ds: Any):
        try:
            if ds is None:
                return
            if self.sensitivity is None:
                self.sensitivity = ds
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
        :param keep_alloc: Keep the sensitivity allocation intact?
        :return: self
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
        :param item: Slice indices
        :return: Sliced signal (SignalSlice)
        """
        return SignalSlice(self, item)


class SignalSlice(Signal):
    """ Slice operator for a Signal
    The sliced values are referenced to their original source Signal, such that they can be used and updated in modules.
    This means that updating the values in this SignalSlice changes the data in its source Signal.
    """
    def __init__(self, orig_signal, sl, tag=None):
        self.orig_signal = orig_signal
        self.slice = sl
        self.keep_alloc = False  # Allocation must be False because sensitivity cannot be assigned with [] operator

        # for s in slice:
        if tag is None:
            self.tag = f"{self.orig_signal.tag}[{fmt_slice(self.slice)}]"
        else:
            self.tag = tag

        # Save error string to location where it is initialized
        self._init_loc = get_init_str()

    @property
    def state(self):
        try:
            return None if self.orig_signal.state is None else self.orig_signal.state[self.slice]
        except Exception as e:
            # Possibilities: Unslicable object (TypeError) or Wrong dimensions or out of range (IndexError)
            raise type(e)("SignalSlice.state (getter)" + self._err_str()) from e

    @state.setter
    def state(self, new_state):
        try:
            self.orig_signal.state[self.slice] = new_state
        except Exception as e:
            # Possibilities: Unslicable object (TypeError) or Wrong dimensions or out of range (IndexError)
            raise type(e)("SignalSlice.state (setter)" + self._err_str()) from e

    @property
    def sensitivity(self):
        try:
            return None if self.orig_signal.sensitivity is None else self.orig_signal.sensitivity[self.slice]
        except Exception as e:
            # Possibilities: Unslicable object (TypeError) or Wrong dimensions or out of range (IndexError)
            raise type(e)("SignalSlice.sensitivity (getter)" + self._err_str()) from e

    @sensitivity.setter
    def sensitivity(self, new_sens):
        try:
            if self.orig_signal.sensitivity is None:
                if new_sens is None:
                    return  # Sensitivity doesn't need to be initialized when it is set to None
                try:
                    self.orig_signal.sensitivity = self.orig_signal.state * 0  # Make a new copy with 0 values
                except TypeError:
                    if self.orig_signal.state is None:
                        raise TypeError("Could not initialize sensitivity because state is not set" + self._err_str())
                    else:
                        raise TypeError(f"Could not initialize sensitivity for type \'{type(self.orig_signal.state).__name__}\'")

            if new_sens is None:
                new_sens = 0  # reset() uses this

            self.orig_signal.sensitivity[self.slice] = new_sens
        except Exception as e:
            # Possibilities: Unslicable object (TypeError) or Wrong dimensions or out of range (IndexError)
            raise type(e)("SignalSlice.sensitivity (setter)" + self._err_str()) from e

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


def _check_valid_signal(sig: Any):
    """ Checks if the argument is a valid Signal object
    :param sig: The object to check
    :return: True if it is a valid Signal
    """
    if isinstance(sig, Signal):
        return True
    if all([hasattr(sig, f) for f in ["state", "sensitivity", "add_sensitivity", "reset"]]):
        return True
    raise TypeError(f"Given argument with type \'{type(sig).__name__}\' is not a valid Signal")


def _check_valid_module(mod: Any):
    """ Checks if the argument is a valid Module object
    :param mod: The object to check
    :return: True if it is a valid Module
    """
    if issubclass(type(mod), Module):
        return True
    if hasattr(mod, "response") and hasattr(mod, "sensitivity") and hasattr(mod, "reset"):
        return True
    raise TypeError(f"Given argument with type \'{type(mod).__name__}\' is not a valid Module")


def _check_function_signature(fn, signals):
    """ Checks the function signature against given signal list
    :param fn: response_ or sensitivity_ function of a Module
    :param signals: The signals involved
    """
    min_args, max_args = 0, 0
    callstr = f"{type(fn.__self__).__name__}.{fn.__name__}{inspect.signature(fn)}"
    for s, p in inspect.signature(fn).parameters.items():
        if p.kind == inspect.Parameter.POSITIONAL_ONLY or p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if p.default != inspect.Parameter.empty:
                raise SyntaxError(f"{callstr} must not have default values \"{p}\"")
            min_args += 1
            if max_args >= 0:
                max_args += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            max_args = -1
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            raise SyntaxError(f"{callstr} may not contain keyword arguments \"{p}\"")
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            raise SyntaxError(f"{callstr} may not contain \"{p}\"")
    if len(signals) < min_args:
        raise TypeError(f"Not enough arguments ({len(signals)}) for {callstr}")
    if max_args >= 0 and len(signals) > max_args:
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

    def _err_str(self, init: bool = True, add_signal: bool = True, fn=None):
        str_list = []
        if init:
            str_list.append(f"Module \'{type(self).__name__}\', initialized in {self._init_loc}")
        if add_signal:
            inp_str = "Inputs: " + ", ".join([s.tag if hasattr(s, 'tag') else 'N/A' for s in self.sig_in]) if len(self.sig_in) > 0 else "No inputs"
            out_str = "Outputs: " + ", ".join([s.tag if hasattr(s, 'tag') else 'N/A' for s in self.sig_out]) if len(self.sig_out) > 0 else "No outputs"
            str_list.append(inp_str + " --> " + out_str)
        if fn is not None:
            name = f"{fn.__self__.__class__.__name__}.{fn.__name__}{inspect.signature(fn)}"
            lineno = inspect.getsourcelines(fn)[1]
            filename = inspect.getfile(fn)
            str_list.append(f"Implemented in File \"{filename}\", line {lineno}, in {name}")
        return err_fmt(*str_list)

    # flake8: noqa: C901
    def __init__(self, sig_in: Union[Signal, List[Signal]] = None, sig_out: Union[Signal, List[Signal]] = None,
                 *args, **kwargs):
        # TODO: Reduce complexity of this init
        self._init_loc = get_init_str()

        self.sig_in = _parse_to_list(sig_in)
        self.sig_out = _parse_to_list(sig_out)
        for i, s in enumerate(self.sig_in):
            try:
                _check_valid_signal(s)
            except Exception as e:
                earg0 = e.args[0] if len(e.args) > 0 else ''
                earg1 = e.args[1:] if len(e.args) > 1 else ()
                raise type(e)(f"Invalid input signal #{i+1} - " + str(earg0) + self._err_str(), *earg1) from None

        for i, s in enumerate(self.sig_out):
            try:
                _check_valid_signal(s)
            except Exception as e:
                earg0 = e.args[0] if len(e.args) > 0 else ''
                earg1 = e.args[1:] if len(e.args) > 1 else ()
                raise type(e)(f"Invalid output signal #{i+1} - " + str(earg0) + self._err_str(), *earg1) from None

        try:
            # Call preparation of submodule with remaining arguments
            self._prepare(*args, **kwargs)
        except Exception as e:
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)("_prepare() - " + str(earg0) + self._err_str(fn=self._prepare), *earg1) from e

        try:
            # Check if the signals match _response() signature
            _check_function_signature(self._response, self.sig_in)
        except Exception as e:
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)(str(earg0) + self._err_str(fn=self._response), *earg1) from None

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
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)(str(earg0) + self._err_str(fn=self._sensitivity), *earg1) from None

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
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)("response() - " + str(earg0) + self._err_str(fn=self._response), *earg1) from e

    def __call__(self):
        return self.response()

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
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)("sensitivity() - " + str(earg0) + self._err_str(fn=self._sensitivity), *earg1) from e

    def reset(self):
        """ Reset the state of the sensitivities (they are set to zero or to None) """
        try:
            [s.reset() for s in self.sig_out]
            [s.reset() for s in self.sig_in]
            self._reset()
            return self
        except Exception as e:
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)("reset() - " + str(earg0) + self._err_str(fn=self._reset), *earg1) from e

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

    >> Network(module1, module2, ...)

    >> Network([module1, module2, ...])

    >> Network((module1, module2, ...))

    >> Network([ {type="module1", sig_in=[sig1, sig2], sig_out=[sig3]},
                 {type="module2", sig_in=[sig3], sig_out=[sig4]} ])

    """
    def __init__(self, *args, print_timing=False):
        self._init_loc = get_init_str()
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

            self.print_timing = print_timing
        except Exception as e:
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)(str(earg0) + self._err_str(add_signal=False), *earg1) from None

    def timefn(self, fn):
        start_t = time.time()
        fn()
        print(f"Evaluating {fn} took {time.time() - start_t} s")

    def response(self):
        if self.print_timing:
            [self.timefn(b.response) for b in self.mods]
        else:
            [b.response() for b in self.mods]

    def sensitivity(self):
        [b.sensitivity() for b in reversed(self.mods)]

    def reset(self):
        [b.reset() for b in reversed(self.mods)]

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

    def append(self, *newmods):
        modlist = _parse_to_list(*newmods)

        # Check if the blocks are initialized, else create them
        for i, b in enumerate(modlist):
            try:  # Check validity of modules
                _check_valid_module(b)
            except Exception as e:
                raise type(e)("append() - Trying to append invalid module " + str(e.args[0])
                              + self._err_str(add_signal=False), *e.args[1:]) from None

        # Obtain the internal blocks
        self.mods.extend(modlist)

        # Gather all the input and output signals of the internal blocks
        all_in = set()
        all_out = set()
        [all_in.update(b.sig_in) for b in self.mods]
        [all_out.update(b.sig_out) for b in self.mods]
        in_unique = all_in - all_out

        self.sig_in = _parse_to_list(in_unique)
        try:
            [_check_valid_signal(s) for s in self.sig_in]
        except Exception as e:
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)("append() - Invalid input signals " + str(earg0)
                          + self._err_str(add_signal=False), *earg1) from None
        self.sig_out = _parse_to_list(all_out)
        try:
            [_check_valid_signal(s) for s in self.sig_out]
        except Exception as e:
            earg0 = e.args[0] if len(e.args) > 0 else ''
            earg1 = e.args[1:] if len(e.args) > 1 else ()
            raise type(e)("append() - Invalid output signals " + str(earg0)
                          + self._err_str(add_signal=False), *earg1) from None

        return modlist[-1].sig_out[0] if len(modlist[-1].sig_out) == 1 else modlist[-1].sig_out  # Returns the output signal
