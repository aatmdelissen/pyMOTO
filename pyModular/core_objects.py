import warnings
import sys


class Signal:
    """
    Saves the state data, connects input and outputs of blocks and manages sensitivities

    Initialize using Signal()
    Optional arguments: tag (string)
    Optional keyword arguments: tag=(string)

    >> Signal('x')

    >> Signal(tag='xPhys')

    """
    def __init__(self, tag=None):
        self.tag = tag
        self.state = None
        self.sensitivity = None

    def set_sens(self, ds):
        try:  # TODO Maybe a check if they are conforming??
            if ds is None:
                return
            if self.sensitivity is None:
                self.sensitivity = ds
            else:
                self.sensitivity += ds
        except Exception as e:
            raise type(e)(str(e) + ', in signal %s' % self.tag).with_traceback(sys.exc_info()[2])

    def get_sens(self):
        return self.sensitivity

    def set_state(self, value):
        self.state = value

    def get_state(self):
        return self.state

    def reset(self):
        self.sensitivity = None


def parse_signals(signals):
    """
    Parses single signal to a list of signals
    :param signals: Signal or list of Signal
    :return: list of Signal
    """
    if signals is None:
        return []
    elif isinstance(signals, list):
        for s in signals:
            if not isinstance(s, Signal):
                raise RuntimeError("Entry {} is not a Signal instance".format(s))
        return signals
    elif isinstance(signals, Signal):
        return [signals]

    raise RuntimeError("Entry {} is not a Signal instance".format(signals))


def parse_to_list(var_in):
    """
    Parses inputs to a list
    :param var_in:
    :return: list
    """
    if var_in is None:
        return []
    elif isinstance(var_in, list):
        return var_in
    elif isinstance(var_in, tuple):
        return list(var_in)
    else:
        return [var_in]


class RegisteredClass(object):
    """
    Abstract base class that can keep track of its subclasses and can instantiate them as well, based on their name.
    """

    @classmethod
    def create(cls, sub_type, *args, **kwargs):
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

    def __init__(self, sig_in=None, sig_out=None, *args, **kwargs):
        try:
            self.sig_in = parse_signals(sig_in)
            self.sig_out = parse_signals(sig_out)

            # Call preparation of module or submodule
            self._prepare(*args, **kwargs)
        except Exception as e:
            raise type(e)(str(e) + ', in module %s' % type(self).__name__).with_traceback(sys.exc_info()[2]) from None

    def response(self):
        """
        Calculate the response from sig_in and output this to sig_out
        """
        try:
            # Get all input states
            inp = [s.get_state() for s in self.sig_in]

            # TODO If error is generated: we want to know in which block
            # Calculate the actual response

            output = parse_to_list(self._response(*inp))

            # Check if enough outputs are calculated
            if len(output) != len(self.sig_out):
                raise RuntimeError("Number of responses calculated ({}) is unequal to number of output signals ({}) {}".
                                   format(len(output), len(self.sig_out), type(self)))

            # Update the output signals
            for i, val in enumerate(output):
                if val is not None:
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

            # Check if any of the input sensitivities is set
            if all([s is None for s in sens_in]):
                return

            # Calculate the new sensitivities of the inputs
            sens_out = parse_to_list(self._sensitivity(*sens_in))

            # Check if enough sensitivities are calculated
            if len(sens_out) != len(self.sig_in):
                raise ValueError("Number of sensitivities calculated ({}) is unequal to number of input signals ({}) {}"
                                 .format(len(sens_out), len(self.sig_in), type(self)))

            # Add the sensitivities to the signals
            for i, ds in enumerate(sens_out):
                self.sig_in[i].set_sens(ds)

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
        raise RuntimeError("No response behavior defined")

    def _sensitivity(self, *args):
        return [None for s in self.sig_in]

    def _reset(self):
        pass


class Interconnection(Module):
    """
    Binds multiple Modules together as one Module

    >> Interconnection(module1, module2, ...)

    >> Interconnection([module1, module2, ...])

    >> Interconnection((module1, module2, ...))

    >> Interconnection([{type="module1", sig_in=[sig1, sig2], sig_out=[sig3]}, {type="module2", sig_in=[sig3], sig_out=[sig4]}])

    """
    def __init__(self, *args):
        # Obtain the internal blocks
        self.mods = parse_to_list(args)

        # Check if the blocks are initialized, else create them
        for i, b in enumerate(self.mods):
            if isinstance(b, dict):
                self.mods[i] = Module.create(b['type'], **b)

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



