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
            raise type(e)(str(e) + ', in signal %s' % self.tag).with_traceback(sys.exc_info()[2]) from None

    def get_sens(self):
        return self._sensitivity

    def reset(self):
        self._sensitivity = None


def make_signals(*args):
    return [Signal(s) for s in args]


def _parse_to_list(*args: Any):
    if len(args) == 0:
        return []
    elif len(args) == 1:
        var_in = args[0]
    else:
        var_in = args

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


class DyadCarrier:
    """ Sparse rank-N matrix
    Stores only the vectors instead of creating a rank-N matrix
    A_ij = sum_k u_{ki} v_{kj}
    """
    def __init__(self, u: Iterable = None, v: Iterable = None):
        """
        This is a class for efficient calculation with dyads / outer products.
        Two vectors u and v can construct a matrix A = uv^T, but this takes a lot of memory to store. Therefore, this
        class only stores the two vectors. Contractions B:uv^T = u^T.B.v can be calculated (

        :param u:
        :param v:
        """

        ulist = _parse_to_list(u)
        vlist = ulist if v is None else _parse_to_list(v)

        if len(ulist) != len(vlist):
            raise TypeError("Number of vectors in u ({}) and v({}) should be equal".format(len(ulist), len(vlist)))

        n = len(ulist)
        self.u = [None for _ in range(n)]
        self.v = [None for _ in range(n)]
        ulen = -1
        vlen = -1
        for i, ui, vi in zip(range(n), ulist, vlist):
            if isinstance(ui, np.ndarray):
                if ui.ndim > 2 or ui.ndim < 1:
                    raise TypeError("Numpy arrays of dimension 1 or 2 accepted. Got {} instead.".format(ui.ndim))

                if ulen < 0:
                    ulen = ui.shape[-1]
                else:
                    if ui.shape[-1] != ulen:
                        raise TypeError("Sizes of vectors unequal. {} != {}.".format(ui.shape[-1], ulen))
                self.u[i] = ui.copy()
            else:
                raise TypeError("Vector in dyadcarrier should be np.ndarray. Got {} instead.".format(type(ui)))

            if isinstance(vi, np.ndarray):
                if vi.ndim > 2 or vi.ndim < 1:
                    raise TypeError("Numpy arrays of dimension 1 or 2 accepted. Got {} instead.".format(ui.ndim))

                if vlen < 0:
                    vlen = vi.shape[-1]
                else:
                    if vi.shape[-1] != vlen:
                        raise TypeError("Sizes of vectors unequal. {} != {}.".format(vi.shape[-1], vlen))
                self.v[i] = vi.copy()
            else:
                raise TypeError("Vector in dyadcarrier should be np.ndarray. Got {} instead.".format(type(vi)))

    def __pos__(self):
        return DyadCarrier(self.u, self.v)

    def __neg__(self):
        return DyadCarrier([-uu for uu in self.u], self.v)

    def __iadd__(self, other):
        self.u += [uu.copy() for uu in other.u]
        self.v += [vv.copy() for vv in other.v]
        return self

    def __isub__(self, other):
        self.u += [-uu for uu in other.u]
        self.v += [vv.copy() for vv in other.v]
        return self

    def __add__(self, other):
        return DyadCarrier(self.u, self.v).__iadd__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def copy(self):
        """
        :return: Copied instance
        """
        return DyadCarrier(self.u, self.v)

    def conj(self):
        """
        :return: Complex conjugated dyad
        """
        return DyadCarrier([u.conj() for u in self.u], [v.conj() for v in self.v])

    def contract(self, mat, rows=None, cols=None):
        """
        Performs contraction using a submatrix

        :param rows: Indices for the rows
        :param cols: Indices for the columns to use
        :param mat_sub: The sub matrix to multiply it with
        :return: Value of the contraction
        """
        val = 0.0
        if rows is None:
            for ui, vi in zip(self.u, self.v):
                val += (ui * mat.dot(vi)).sum()
        elif rows.ndim == 1:
            for ui, vi in zip(self.u, self.v):
                val += (ui[rows] * mat.dot(vi[cols])).sum()
        elif rows.ndim == 2:
            val = np.zeros(rows.shape[0])
            if rows.shape == mat.shape:
                # Unstructured mode
                for ui, vi in zip(self.u, self.v):
                    val += np.sum(ui[rows] * mat * vi[cols], axis=1)
            else:
                # Structured mode
                for ui, vi in zip(self.u, self.v):
                    val += (np.dot(ui[rows], mat) * vi[cols]).sum(1)

        return val

    def expand(self):
        nrows = self.u[0].shape[0]
        ncols = self.v[0].shape[0]
        if max(nrows, ncols) > 1000:
            raise RuntimeWarning("Expanding a dyad results into a dense matrix. "
                                 "This is not advised for large matrices: {}x{}".format(nrows, ncols))
        if self.iscomplex():
            val = np.zeros((nrows, ncols), dtype='complex128')
        else:
            val = np.zeros((nrows, ncols))

        for ui, vi in zip(self.u, self.v):
            val += np.outer(ui, vi)
        return val

    def iscomplex(self):
        """ Check if the DyadCarrier is complex """
        for ui in self.u:
            if np.iscomplexobj(ui):
                return True

        for vi in self.v:
            if np.iscomplexobj(vi):
                return True

        return False


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
