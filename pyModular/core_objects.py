from typing import Union, List, Iterable, Any
import warnings
import sys
import numpy as np


class Signal:
    """
    Saves the state data, connects input and outputs of blocks and manages sensitivities

    Initialize using Signal()
    Optional arguments: tag (string)
    Optional keyword arguments: tag=(string)

    >> Signal('x1')

    >> Signal(tag='x2')

    """
    def __init__(self, tag: str = ""):
        self.tag = tag
        self._state = None
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
            raise type(e)(str(e) + ', in signal %s' % self.tag).with_traceback(sys.exc_info()[2])

    def get_sens(self):
        return self._sensitivity

    def reset(self):
        self._sensitivity = None


def make_signals(*args):
    return [Signal(s) for s in args]


def _parse_to_list(var_in: Any):
    if var_in is None:
        return []
    elif isinstance(var_in, np.ndarray):
        return [var_in]
    elif isinstance(var_in, Iterable):
        return list(var_in)
    else:
        return [var_in]


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
        subs = dict((k.lower(), v) for k, v in cls.all_subclasses().items())
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
        for sc in subs:
            scn = sc.__name__.lower()
            if scn not in seen:
                seen[scn] = 1
                subs_out[sc.__name__] = sc
            else:
                if seen[scn] == 1:
                    duplicates.append(scn)
                seen[scn] += 1

        # Emit warning if duplicates are found
        if duplicates:
            msg = "\n ".join([" - '{}' currently defined as {}".format(d, subs_out[d]) for d in duplicates])
            warnings.warn("Duplicate blocks defined:\n{}".format(msg), Warning)

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
                raise RuntimeError("Number of responses calculated ({}) is unequal to number of output signals ({}) {}".
                                   format(len(state_out), len(self.sig_out), type(self)))

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
            # Check if the block has outputs, if not, no sensitivity can be calculated
            if len(self.sig_out) == 0:
                return

            # Get all sensitivity values of the outputs
            sens_in = [s.get_sens() for s in self.sig_out]

            if all([s is None for s in sens_in]):
                return  # If none of the adjoint variables is set

            # Calculate the new sensitivities of the inputs
            sens_out = _parse_to_list(self._sensitivity(*sens_in))

            # Check if enough sensitivities are calculated
            if len(sens_out) != len(self.sig_in):
                raise ValueError("Number of sensitivities calculated ({}) is unequal to number of input signals ({}) {}"
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
        # Obtain the internal blocks
        self.mods = _parse_to_list(args)

        # Check if the blocks are initialized, else create them
        for i, b in enumerate(self.mods):
            if isinstance(b, dict):
                self.mods[i] = Module.create(b['type'], **b)

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

    def response(self):
        [b.response() for b in self.mods]

    def sensitivity(self):
        [b.sensitivity() for b in reversed(self.mods)]

    def reset(self):
        [b.reset() for b in self.mods]

    def _response(self, *args):
        pass  # Unused
