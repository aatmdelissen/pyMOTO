from typing import Union, List, Any
import warnings
import sys
from .utils import _parse_to_list


class Signal:
    """
    Saves the state data, connects input and outputs of blocks and manages sensitivities

    Initialize using Signal()
    Optional arguments: tag (string)
    Optional keyword arguments: tag=(string)

    >> Signal('x1')

    >> Signal(tag='x2')

    """
    def __init__(self, tag: str = "", state: Any = None):
        self.tag = tag
        self._state = state
        self._sensitivity = None

    def set_state(self, value: Any):
        self._state = value

    def get_state(self):
        return self._state

    def set_sens(self, ds: Any):
        self._sensitivity = ds

    def add_sens(self, ds: Any):
        try:
            if ds is None:
                return
            if self._sensitivity is None:
                self.set_sens(ds)
            else:
                self._sensitivity += ds
        except Exception as e:
            raise type(e)(str(e) + ', in signal %s' % self.tag).with_traceback(sys.exc_info()[2]) from None

    def get_sens(self):
        return self._sensitivity

    def reset(self):
        self._sensitivity = None


def make_signals(*args):
    return [Signal(s) for s in args]


def _check_valid_signal(sig: Any):
    if not isinstance(sig, Signal):
        raise TypeError("Entry {} is not a Signal".format(sig))


def _check_valid_module(mod: Any):
    if not issubclass(type(mod), Module):
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


class Module(RegisteredClass):
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
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def response(self):
        """
        Calculate the response from sig_in and output this to sig_out
        """
        try:
            inp = [s.get_state() for s in self.sig_in]
            state_out = _parse_to_list(self._response(*inp))  # Calculate the actual response

            # Check if enough outputs are calculated
            if len(state_out) != len(self.sig_out):
                raise TypeError("Number of responses calculated ({}) is unequal to number of output signals ({}) {}"
                                .format(len(state_out), len(self.sig_out), type(self)))

            # Update the output signals
            for i, val in enumerate(state_out):
                self.sig_out[i].set_state(val)

        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def sensitivity(self):
        """
        Based on the sensitivity we get from sig_out, reverse the process and output the new sensitivities to sig_in
        """
        try:
            # Get all sensitivity values of the outputs
            sens_in = [s.get_sens() for s in self.sig_out]

            if len(self.sig_out) > 0 and all([s is None for s in sens_in]):
                return  # If none of the adjoint variables is set

            # Calculate the new sensitivities of the inputs
            sens_out = _parse_to_list(self._sensitivity(*sens_in))

            # Check if enough sensitivities are calculated
            if len(sens_out) != len(self.sig_in):
                raise TypeError("Number of sensitivities calculated ({}) is unequal to number of input signals ({}) {}"
                                .format(len(sens_out), len(self.sig_in), type(self)))

            # Add the sensitivities to the signals
            for i, ds in enumerate(sens_out):
                self.sig_in[i].add_sens(ds)

        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def reset(self):
        try:
            [s.reset() for s in self.sig_out]
            [s.reset() for s in self.sig_in]
            self._reset()
        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    # METHODS TO BE DEFINED BY USER
    def _prepare(self, *args, **kwargs):
        pass

    def _response(self, *args):
        raise NotImplementedError("No response behavior defined")

    def _sensitivity(self, *args):
        if len(self.sig_out) > 0 and len(self.sig_in) > 0:
            warnings.warn("Sensitivity routine is used, but not defined, in {}".format(type(self).__name__), Warning)
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
        try:
            [b.response() for b in self.mods]
        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def sensitivity(self):
        try:
            [b.sensitivity() for b in reversed(self.mods)]
        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def reset(self):
        try:
            [b.reset() for b in self.mods]
        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def _response(self, *args):
        pass  # Unused
