import numpy as np
from .utils import _parse_to_list, _concatenate_to_array
from .core_objects import Signal, SignalSlice, Module, Network
from .common.mma import MMA
from typing import List, Iterable, Union, Callable


def _has_signal_overlap(sig1: List[Signal], sig2: List[Signal]):
    for s1 in sig1:
        while isinstance(s1, SignalSlice):
            s1 = s1.orig_signal
        for s2 in sig2:
            while isinstance(s2, SignalSlice):
                s2 = s2.orig_signal
            if s1 == s2:
                return True
    return False


# flake8: noqa: C901
def finite_difference(blk: Module, fromsig: Union[Signal, Iterable[Signal]] = None,
                      tosig: Union[Signal, Iterable[Signal]] = None,
                      dx: float = 1e-8, relative_dx: bool = False, tol: float = 1e-5, random: bool = True,
                      use_df: list = None, test_fn: Callable = None, keep_zero_structure=True, verbose=True):
    """ Performs a finite difference check on the given Module or Network

    Args:
        blk: The module or network
        fromsig (optional): Specify input signals of interest
        tosig (optional): Specify output signals of interest

    Keyword Args:
        dx: Perturbation size
        relative_dx: Use a relative perturbation size or not
        tol: Tolerance
        random: Randomize sensitivity data
        use_df: Give pre-defined sensitivity data
        test_fn: A generic test function (x, dx, df_an, df_fd)
        keep_zero_structure: If ``True`` variables that are ``0`` are not perturbed
        verbose: Print extra information to console
    """
    print("=========================================================================================================\n")
    print("Starting finite difference of \"{0}\" with dx = {1}, and tol = {2}".format(type(blk).__name__, dx, tol))

    # Parse inputs and outputs to be finite-differenced
    inps = blk.sig_in if fromsig is None else _parse_to_list(fromsig)
    outps = blk.sig_out if tosig is None else _parse_to_list(tosig)

    # In case a Network is passed, only the blocks connecting input and output need execution
    if isinstance(blk, Network):
        i_first, i_last = -1, -1
        for i, b in enumerate(blk.mods):
            # Find the first module which requires any of the input signals
            if i_first < 0 and _has_signal_overlap(inps, b.sig_in):
                i_first = i
            # Find the last module that generates any of the output signals
            if _has_signal_overlap(outps, b.sig_out):
                i_last = i
        if i_first < 0:
            raise RuntimeError("Could not find any modules that use any of the provided input signals")
        if i_last < 0:
            raise RuntimeError("Could not find any modules that use any of the provided output signals")
        blks_pre = Network(blk.mods[:i_first])
        blk = Network(blk.mods[i_first:i_last+1])
        # Precompute only once for any blocks before first occurrence of <inps>
        blks_pre.response()

    print("Inputs:")
    if verbose:
        [print("{}\t{} = {}".format(i, s.tag, s.state)) for i, s in enumerate(inps)]
    else:
        print(", ".join([s.tag for s in inps]))
    print("")

    # Setup some internal storage
    f0 = [None for _ in outps]  # Response values at original inputs
    df_an = [np.empty(0) for _ in outps]  # Analytical output sensitivities
    dx_an = [[np.empty(0) for _ in inps] for _ in outps]

    # Initial reset in case some memory is still left
    blk.reset()

    # Perform response
    blk.response()

    print("Outputs:")
    if not verbose:
        print(", ".join([s.tag for s in outps]))

    # Get analytical response and sensitivities, by looping over all outputs
    for Iout, Sout in enumerate(outps):
        # Obtain the output state
        output = Sout.state
        if verbose:
            print("{}\t{} = {}".format(Iout, Sout.tag, output))

        # Store the output value
        f0[Iout] = (output.copy() if hasattr(output, "copy") else output)

        # Get the output state shape
        shape = (output.shape if hasattr(output, "shape") else ())

        # Generate a (random) sensitivity for output signal
        if use_df is not None:
            df_an[Iout] = use_df[Iout]
        else:
            if random:
                df_an[Iout] = np.random.rand(*shape)
            else:
                df_an[Iout] = np.ones(shape)

            if np.iscomplexobj(output):
                if random:
                    df_an[Iout] = df_an[Iout] + 1j * np.random.rand(*shape)
                else:
                    df_an[Iout] = df_an[Iout] + 1j * np.ones(shape)

        # Set the output sensitivity
        Sout.sensitivity = df_an[Iout]

        # Perform the analytical sensitivity calculation
        blk.sensitivity()

        # Store all input sensitivities for this output
        for Iin, Sin in enumerate(inps):
            sens = Sin.sensitivity
            dx_an[Iout][Iin] = (sens.copy() if hasattr(sens, "copy") else sens)

        # Reset the sensitivities for next output
        blk.reset()

    # Perturb each of the input signals
    for Iin, Sin in enumerate(inps):
        print("___________________________________________________")
        print("Perturbing input {} \"{}\"...\n".format(Iin, Sin.tag))

        # Get input state
        x = Sin.state

        try:
            # Get iterator for x
            it = np.nditer(x, flags=['c_index', 'multi_index'], op_flags=['readwrite'])
            is_iterable = True
        except TypeError:
            it = np.nditer(np.array(x), flags=['c_index', 'multi_index'], op_flags=['readwrite'])
            is_iterable = False

        i_failed, i_tested = 0, 0
        # Loop over all values in x
        while not it.finished:
            # Get original value and do the perturbation
            if is_iterable:
                x0 = it[0].copy()
                if x0 == 0 and keep_zero_structure:
                    it.iternext()
                    continue
                sf = np.abs(x0) if (relative_dx and np.abs(x0)!=0) else 1.0  # Scale factor
                it[0] += dx*sf
                Sin.state = x
            else:
                x0 = it[0].item()
                sf = np.abs(x0) if (relative_dx and np.abs(x0) != 0) else 1.0
                Sin.state = x0 + dx*sf

            # Calculate perturbed solution
            blk.response()

            # Obtain all perturbed responses
            for Iout, Sout in enumerate(outps):
                # Obtain perturbed response
                fp = Sout.state

                # Finite difference sensitivity
                df = (fp - f0[Iout])/(dx*sf)

                dgdx_fd = np.real(np.sum(df*df_an[Iout]))

                if dx_an[Iout][Iin] is not None:
                    try:
                        dgdx_an = np.real(dx_an[Iout][Iin][it.multi_index])
                    except (IndexError, TypeError):
                        dgdx_an = np.real(dx_an[Iout][Iin])
                else:
                    dgdx_an = 0.0

                if abs(dgdx_an) < tol:
                    error = abs(dgdx_fd - dgdx_an)
                else:
                    error = abs(dgdx_fd - dgdx_an)/max(abs(dgdx_fd), abs(dgdx_an))

                i_tested += 1
                if error > tol:
                    i_failed += 1

                if verbose or error > tol:
                    print("δ%s/δ%s     i = %s \tAn :% .3e \tFD : % .3e \tError: % .3e %s"
                          % (Sout.tag, Sin.tag, it.multi_index, dgdx_an, dgdx_fd, error, "<--*" if error > tol else ""))

                if test_fn is not None:
                    test_fn(x0, dx, dgdx_an, dgdx_fd)

            # Restore original state
            if is_iterable:
                it[0] = x0
                Sin.state = x
            else:
                Sin.state = x0

            # If the input state is complex, also do a perturbation in the imaginary direction
            if np.iscomplexobj(x0):
                # Do the perturbation
                if is_iterable:
                    it[0] += dx*1j*sf
                else:
                    Sin.state = x0 + dx*1j*sf

                # Calculate perturbed solution
                blk.response()

                # Obtain all perturbed responses
                for Iout, Sout in enumerate(outps):
                    # Obtain perturbed response
                    fp = Sout.state

                    # Finite difference sensitivity
                    df = (fp - f0[Iout])/(dx*1j*sf)
                    dgdx_fd = np.imag(np.sum(df*df_an[Iout]))

                    if dx_an[Iout][Iin] is not None:
                        try:
                            dgdx_an = np.imag(dx_an[Iout][Iin][it.multi_index])
                        except IndexError:
                            dgdx_an = np.imag(dx_an[Iout][Iin])
                    else:
                        dgdx_an = 0.0

                    if abs(dgdx_an) < tol:
                        error = abs(dgdx_fd - dgdx_an)
                    else:
                        error = abs(dgdx_fd - dgdx_an)/max(abs(dgdx_fd), abs(dgdx_an))

                    i_tested += 1
                    if error > tol:
                        i_failed += 1

                    if verbose or error > tol:
                        print("δ%s/δ%s (I) i = %s \tAn :% .3e \tFD : % .3e \tError: % .3e %s"
                              % (Sout.tag, Sin.tag, it.multi_index, dgdx_an, dgdx_fd, error, "<--*" if error > tol else ""))

                    if test_fn is not None:
                        test_fn(x0, dx, dgdx_an, dgdx_fd)

                # Restore original state
                if is_iterable:
                    it[0] = x0
                else:
                    Sin.state = x0

            # Go to the next entry in the array
            it.iternext()

        print(f"-- Number of finite difference values beyond tolerance ({tol}) = {i_failed} / {i_tested}")

    print("___________________________________________________")
    print(f"\nFinished finite-difference check of \"{type(blk).__name__}\"\n")
    print("=========================================================================================================\n")


def obtain_sensitivities(signals: Iterable[Signal]) -> List:
    """ Obtains sensitivities from a list of signals, replacing None by zeros of correct length

    :param signals: The list of signals
    :return: List of sensitivities
    """
    sens = []

    for s in signals:
        the_sens = s.sensitivity

        if the_sens is None:
            the_state = s.state
            if the_state is not None:
                the_sens = np.zeros_like(the_state)

        sens.append(the_sens)
    return sens


def minimize_oc(function, variables, objective: Signal,
                tolx=1e-4, tolf=1e-4, maxit=100, xmin=0.0, xmax=1.0, move=0.2,
                l1init=0, l2init=100000, l1l2tol=1e-4, maxvol=None, verbosity=2):
    """ Execute minimization using the OC-method

    Args:
        function: The Network defining the optimization problem
        variables: The Signals defining the design variables
        objective: The objective function Signal to be minimized

    Keyword Args:
        tolx: Stopping criterium for relative design change
        tolf: Stopping criterium for relative objective change
        maxit: Maximum number of iteration
        xmin: Minimum design variable (can be a vector)
        xmax: Maximum design variable (can be a vector)
        move: Move limit
        l1init: OC internal parameter
        l2init: OC internal parameter
        l1l2tol: OC internal parameter
        verbosity: 0 - No prints, 1 - Only convergence message, 2 - Convergence and iteration info

    """
    variables = _parse_to_list(variables)
    xval, cumlens = _concatenate_to_array([s.state for s in variables])

    if maxvol is None:
        maxvol = np.sum(xval)

    f = 0.0
    for it in range(maxit):
        # Calculate response
        function.response()
        fprev, f = f, objective.state
        rel_fchange = abs(f-fprev)/abs(f)
        if rel_fchange < tolf:
            if verbosity >= 1:
                print(f"OC converged: Relative function change |Δf|/|f| ({rel_fchange}) below tolerance ({tolf})")
            break

        if verbosity >= 2:
            print("It. {0: 4d}, f0 = {1: .2e}, Δf = {2: .2e}".format(it, f, f-fprev))

        # Calculate sensitivity of the objective
        function.reset()
        objective.sensitivity = 1.0
        function.sensitivity()
        dfdx, _ = _concatenate_to_array(obtain_sensitivities(variables))
        maxdfdx = max(dfdx)
        if maxdfdx > 0:
            raise RuntimeError(f"OC only works for negative sensitivities: max(dfdx) = {maxdfdx}")

        # Do OC update
        l1, l2 = l1init, l2init
        while l2 - l1 > l1l2tol:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(xval * np.sqrt(-dfdx / lmid), np.maximum(xmin, xval-move), np.minimum(xmax, xval+move))
            l1, l2 = (lmid, l2) if np.sum(xnew) - maxvol > 0 else (l1, lmid)

        # Stopping criteria on step size
        rel_stepsize = np.linalg.norm(xval - xnew)/np.linalg.norm(xval)
        if rel_stepsize < tolx:
            if verbosity >= 1:
                print(f"OC converged: Relative stepsize |Δx|/|x| ({rel_stepsize}) below tolerance ({tolx})")
            break

        xval = xnew
        # Set the new states
        for i, s in enumerate(variables):
            s.state = xnew[cumlens[i]:cumlens[i + 1]]


def minimize_mma(function, variables, responses, **kwargs):
    """ Execute minimization using the MMA-method
    Svanberg (1987), The method of moving asymptotes - a new method for structural optimization

    Args:
        function: The Network defining the optimization problem
        variables: The Signals defining the design variables
        responses: A list of Signals, where the first is to be minimized and the others are constraints.

    Keyword Args:
        tolx: Stopping criterium for relative design change
        tolf: Stopping criterium for relative objective change
        maxit: Maximum number of iteration
        move: Move limit on relative variable change per iteration
        xmin: Minimum design variable (can be a vector)
        xmax: Maximum design variable (can be a vector)
        verbosity: 0 - No prints, 1 - Only convergence message, 2 - Convergence and iteration info, 3 - Extended info

    """
    # Save initial state
    mma = MMA(function, variables, responses, **kwargs)
    mma.response()
