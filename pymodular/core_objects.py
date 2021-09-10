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
        self.tag = tag
        self.state = state
        self.sensitivity = sensitivity

    def add_sensitivity(self, ds: Any):
        try:
            if ds is None:
                return
            if self.sensitivity is None:
                self.sensitivity = ds
            else:
                self.sensitivity += ds
        except Exception as e:
            raise type(e)(str(e) + ', in signal %s' % self.tag).with_traceback(sys.exc_info()[2]) from None

    def reset(self, keep_alloc: bool = False):
        if keep_alloc and self.sensitivity is not None:
            self.sensitivity *= 0
        else:
            self.sensitivity = None


def make_signals(*args):
    return [Signal(s) for s in args]


def _check_valid_signal(sig: Any):
    if isinstance(sig, Signal):
        return True
    if hasattr(sig, "state") and hasattr(sig, "sensitivity"):
        return True
    if hasattr(sig, "add_sensitivity"):
        return True
    raise TypeError("Entry {} is not a Signal".format(sig))


def _check_valid_module(mod: Any):
    if issubclass(type(mod), Module):
        return True
    if hasattr(mod, "response") and hasattr(mod, "sensitivity") and hasattr(mod, "reset"):
        return True
    raise TypeError("Entry {} is not a Module".format(mod))


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

    def _error_str(self, add_signal: bool=True):
        sig_str = ""
        if add_signal:
            sig_str = f"{[s.tag for s in self.sig_in]}->{[s.tag for s in self.sig_out]}"
        lineno = inspect.getsourcelines(self.__class__)[1]
        filename = inspect.getfile(self.__class__)
        return f"\n\tFile \"{filename}\", line {lineno}, in module {type(self).__name__}{sig_str}"

    def __init__(self, sig_in: Union[Signal, List[Signal]] = None, sig_out: Union[Signal, List[Signal]] = None,
                 *args, **kwargs):
        try:
            self.sig_in = _parse_to_list(sig_in)
            [_check_valid_signal(s) for s in self.sig_in]

            self.sig_out = _parse_to_list(sig_out)
            [_check_valid_signal(s) for s in self.sig_out]

            # Call preparation of submodule with remaining arguments
            self._prepare(*args, **kwargs)
        except Exception as e:
            raise type(e)(str(e) + self._error_str(add_signal=False)).with_traceback(sys.exc_info()[2]) from None

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
            raise type(e)(str(e) + self._error_str()).with_traceback(sys.exc_info()[2]) from None

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
            raise type(e)(str(e) + self._error_str()).with_traceback(sys.exc_info()[2]) from None

    def reset(self):
        try:
            [s.reset() for s in self.sig_out]
            [s.reset() for s in self.sig_in]
            self._reset()
        except Exception as e:
            raise type(e)(str(e) + self._error_str()).with_traceback(sys.exc_info()[2]) from None

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
        for b in self.mods:
            try:
                b.response()
            except Exception as e:
                raise type(e)(str(e) + f', in module {type(self).__name__}' + b._error_str()).with_traceback(sys.exc_info()[2]) from None

    def sensitivity(self):
        try:
            [b.sensitivity() for b in reversed(self.mods)]
        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def reset(self):
        try:
            [b.reset() for b in reversed(self.mods)]
        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def _response(self, *args):
        pass  # Unused

    def append(self, *newmods):
        # Obtain the internal blocks
        self.mods.append(*_parse_to_list(*newmods))

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

