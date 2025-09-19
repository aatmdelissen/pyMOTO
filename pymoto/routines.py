import warnings
import numpy as np
from .utils import _parse_to_list, _concatenate_to_array
from .core_objects import Signal, Module, Network, SignalsT
from .common.mma import MMA
from typing import List, Iterable, Callable
from scipy.sparse import issparse
from scipy.optimize import linprog


# flake8: noqa: C901
def finite_difference(
    fromsig: SignalsT = None,
    tosig: SignalsT = None,
    function: Module = None,
    dx: float = 1e-8,
    relative_dx: bool = False,
    tol: float = 1e-5,
    random: bool = True,
    use_df: list = None,
    test_fn: Callable = None,
    keep_zero_structure=True,
    verbose=True,
):
    """Performs a finite difference check on the given Module or Network

    Args:
        fromsig (optional): Specify input signals of interest
        tosig (optional): Specify output signals of interest
        function (optional): The module or network

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
    if function is None:  # Default grab the global network, if none is provided
        function = Network.active[0]

    print("\n=========================================================================================================")
    print(f'Starting finite difference of "{type(function).__name__}" with dx = {dx}, and tol = {tol}')

    # In case a Network is passed, only the blocks connecting input and output need execution
    if isinstance(function, Network):
        subfn = function.get_output_cone(tosig).get_input_cone(fromsig)
        if len(subfn) == 0:
            raise RuntimeError(
                f"Could not find a network that use the provided input signals {fromsig} "
                f"and produce the requested output signals {tosig}"
            )
        inps = subfn.sig_in if fromsig is None else _parse_to_list(fromsig)
        outps = subfn.sig_out if tosig is None else _parse_to_list(tosig)

        # Check if all required states have a value, else try to run anything up to input signals
        if any([s.state is None for s in inps]):
            blks_pre = function.get_output_cone(tosig=inps)
            blks_pre.response()

        function = subfn
    else:  # Module
        # Parse inputs and outputs to be finite-differenced
        inps = function.sig_in if fromsig is None else _parse_to_list(fromsig)
        outps = function.sig_out if tosig is None else _parse_to_list(tosig)

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
    function.reset()

    # Perform response
    function.response()

    print("Outputs:")
    if verbose:
        [print("{}\t{} = {}".format(i, s.tag, s.state)) for i, s in enumerate(outps)]
    else:
        print(", ".join([s.tag for s in outps]))

    # Get analytical response and sensitivities, by looping over all outputs
    for Iout, Sout in enumerate(outps):
        # Obtain the output state
        output = Sout.state

        # Store the output value
        f0[Iout] = output.copy() if hasattr(output, "copy") else output

        # Get the output state shape
        shape = output.shape if hasattr(output, "shape") else ()

        if output is None:
            warnings.warn(f"Output {Iout} of {Sout.tag} is None")
            continue

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
        function.sensitivity()

        # Store all input sensitivities for this output
        for Iin, Sin in enumerate(inps):
            sens = Sin.sensitivity
            dx_an[Iout][Iin] = sens.copy() if hasattr(sens, "copy") else sens

        # Reset the sensitivities for next output
        function.reset()

    # Perturb each of the input signals
    for Iin, Sin in enumerate(inps):
        print("___________________________________________________")
        print('Perturbing input {} "{}"...\n'.format(Iin, Sin.tag))

        # Get input state
        x = Sin.state

        try:
            # Get iterator for x
            it = np.nditer(x, flags=["c_index", "multi_index"], op_flags=["readwrite"])
            is_iterable = True
        except TypeError:
            it = np.nditer(np.array(x), flags=["c_index", "multi_index"], op_flags=["readwrite"])
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
                sf = np.abs(x0) if (relative_dx and np.abs(x0) != 0) else 1.0  # Scale factor
                it[0] += dx * sf
                Sin.state = x
            else:
                x0 = it[0].item()
                sf = np.abs(x0) if (relative_dx and np.abs(x0) != 0) else 1.0
                Sin.state = x0 + dx * sf

            # Calculate perturbed solution
            function.response()

            # Obtain all perturbed responses
            for Iout, Sout in enumerate(outps):
                # Obtain perturbed response
                fp = Sout.state
                if fp is None:
                    warnings.warn(f"Output {Iout} of {Sout.tag} is None")
                    continue

                # Finite difference sensitivity
                if issparse(fp):
                    df = (fp.toarray() - f0[Iout].toarray()) / (dx * sf)
                else:
                    df = (fp - f0[Iout]) / (dx * sf)

                dgdx_fd = np.real(np.sum(df * df_an[Iout]))

                if dx_an[Iout][Iin] is not None:
                    try:
                        dgdx_an = np.real(dx_an[Iout][Iin][it.multi_index])
                    except (IndexError, TypeError):
                        dgdx_an = np.real(dx_an[Iout][Iin])
                else:
                    dgdx_an = 0.0

                if abs(dgdx_an) == 0:
                    error = abs(dgdx_fd - dgdx_an)
                else:
                    error = abs(dgdx_fd - dgdx_an) / max(abs(dgdx_fd), abs(dgdx_an))

                i_tested += 1
                if error > tol:
                    i_failed += 1

                if verbose or error > tol:
                    print(
                        "δ%s/δ%s     i = %s \tAn :% .3e \tFD : % .3e \tError: % .3e %s"
                        % (Sout.tag, Sin.tag, it.multi_index, dgdx_an, dgdx_fd, error, "<--*" if error > tol else "")
                    )

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
                    it[0] += dx * 1j * sf
                else:
                    Sin.state = x0 + dx * 1j * sf

                # Calculate perturbed solution
                function.response()

                # Obtain all perturbed responses
                for Iout, Sout in enumerate(outps):
                    # Obtain perturbed response
                    fp = Sout.state

                    if fp is None:
                        warnings.warn(f"Output {Iout} of {Sout.tag} is None")
                        continue

                    # Finite difference sensitivity
                    if issparse(fp):
                        df = (fp.toarray() - f0[Iout].toarray()) / (dx * 1j * sf)
                    else:
                        df = (fp - f0[Iout]) / (dx * 1j * sf)
                    dgdx_fd = np.imag(np.sum(df * df_an[Iout]))

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
                        error = abs(dgdx_fd - dgdx_an) / max(abs(dgdx_fd), abs(dgdx_an))

                    i_tested += 1
                    if error > tol:
                        i_failed += 1

                    if verbose or error > tol:
                        print(
                            "δ%s/δ%s (I) i = %s \tAn :% .3e \tFD : % .3e \tError: % .3e %s"
                            % (
                                Sout.tag,
                                Sin.tag,
                                it.multi_index,
                                dgdx_an,
                                dgdx_fd,
                                error,
                                "<--*" if error > tol else "",
                            )
                        )

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
    print(f'\nFinished finite-difference check of "{type(function).__name__}"\n')
    print("=========================================================================================================\n")


def obtain_sensitivities(signals: Iterable[Signal]) -> List:
    """Obtains sensitivities from a list of signals, replacing None by zeros of correct length

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


def minimize_oc(
    variables: SignalsT,
    objective: Signal,
    function: Module = None,
    tolx=1e-4,
    tolf=1e-4,
    maxit=100,
    xmin=0.0,
    xmax=1.0,
    move=0.2,
    l1init=0,
    l2init=100000,
    l1l2tol=1e-4,
    maxvol=None,
    verbosity=2,
):
    """Execute minimization using the OC-method

    Args:
        variables: The Signals defining the design variables
        objective: The objective function Signal to be minimized
        function (optional): The Network defining the optimization problem

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

    if function is None:
        function = Network.active[0]

    if maxvol is None:
        maxvol = np.sum(xval)

    f = 0.0
    for it in range(maxit):
        # Calculate response
        if it > 0 or objective.state is None:
            function.response()
        fprev, f = f, objective.state
        rel_fchange = abs(f - fprev) / abs(f)
        if rel_fchange < tolf:
            if verbosity >= 1:
                print(f"OC converged: Relative function change |Δf|/|f| ({rel_fchange}) below tolerance ({tolf})")
            break

        if verbosity >= 2:
            print("It. {0: 4d}, f0 = {1: .2e}, Δf = {2: .2e}".format(it, f, f - fprev))

        # Calculate sensitivity of the objective
        function.reset()
        objective.sensitivity = 1.0
        function.sensitivity()
        dfdx, _ = _concatenate_to_array(obtain_sensitivities(variables))
        maxdfdx = max(dfdx)
        if maxdfdx > 1e-15:
            warnings.warn(f"OC only works for negative sensitivities: max(dfdx) = {maxdfdx}. Clipping positive values.")
        dfdx = np.minimum(dfdx, 0)

        # Do OC update
        l1, l2 = l1init, l2init
        while l2 - l1 > l1l2tol:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(xval * np.sqrt(-dfdx / lmid), np.maximum(xmin, xval - move), np.minimum(xmax, xval + move))
            l1, l2 = (lmid, l2) if np.sum(xnew) - maxvol > 0 else (l1, lmid)

        # Stopping criteria on step size
        rel_stepsize = np.linalg.norm(xval - xnew) / np.linalg.norm(xval)
        if rel_stepsize < tolx:
            if verbosity >= 1:
                print(f"OC converged: Relative stepsize |Δx|/|x| ({rel_stepsize}) below tolerance ({tolx})")
            break

        xval = xnew
        # Set the new states
        for i, s in enumerate(variables):
            s.state = xnew[cumlens[i] : cumlens[i + 1]]


def minimize_mma(variables: SignalsT, responses: SignalsT, function: Module = None, **kwargs):
    """Execute minimization using the MMA-method
    Svanberg (1987), The method of moving asymptotes - a new method for structural optimization

    Args:
        variables: The Signals defining the design variables
        responses: A list of Signals, where the first is to be minimized and the others are constraints
        function (optional): The Network defining the optimization problem

    Keyword Args:
        tolx: Stopping criterium for relative design change
        tolf: Stopping criterium for relative objective change
        maxit: Maximum number of iteration
        move: Move limit on relative variable change per iteration
        xmin: Minimum design variable (can be a vector)
        xmax: Maximum design variable (can be a vector)
        verbosity: Level of information to print
          0 - No prints
          1 - Only convergence message
          2 - Convergence and iteration info (default)
          3 - Additional info on variables
          4 - Additional info on sensitivity information

    """
    # Save initial state
    if function is None:
        function = Network.active[0]
    mma = MMA(function, variables, responses, **kwargs)
    mma.response()


def minimize_slp(variables, responses, function=None, xmin=0, xmax=1, move=0.2, maxit=100, tolx=1e-4, tolf=1e-4, 
                 verbosity: int = 2, adaptive_movelimit: bool = True):
    """Sequential linear programming optimization algorithm

    Args:
        variables: The Signals defining the design variables
        responses: A list of Signals, where the first is to be minimized and the others are constraints
        function (optional): The Network defining the optimization problem
        xmin (optional): Minimum design variable (can be a vector)
        xmax (optional): Maximum design variable (can be a vector)
        move (optional): Move limit on relative variable change per iteration
        maxit (optional): Maximum number of iteration
        tolx (optional): Stopping criterium for relative design change
        tolf (optional): Stopping criterium for relative objective change
        verbosity (optional): Level of information to print
          0 - No prints
          1 - Only convergence message
          2 - Convergence and iteration info (default)
          3 - Additional info on variables
          4 - Additional info on sensitivity information
        adaptive_movelimit (optional): Move limit is adapted based on variable oscillation behavior
    """
    variables = _parse_to_list(variables)
    responses = _parse_to_list(responses)

    if len(responses) > 1:
        raise NotImplementedError("SLP currently only for unconstrained optimization")

    # For adaptive movelimit
    asyincr = 1.2
    asydecr = 0.7
    asyinit = 1.0
    asybound = 10.0

    # Save initial state
    xval, cumlens = _concatenate_to_array([s.state for s in variables])
    n = len(xval)

    # Set lower bounds
    if not hasattr(xmin, "__len__"):
        xmin = xmin * np.ones_like(xval)
    elif len(xmin) == len(variables):
        xminvals = xmin
        xmin = np.zeros_like(xval)
        for i in range(len(xminvals)):
            xmin[cumlens[i]:cumlens[i + 1]] = xminvals[i]

    if len(xmin) != n:
        raise RuntimeError(f"Length of the xmin vector ({len(xmin)}) should be equal to # design variables ({n})")

    # Upper bounds
    if not hasattr(xmax, "__len__"):
        xmax = xmax * np.ones_like(xval)
    elif len(xmax) == len(variables):
        xmaxvals = xmax
        xmax = np.zeros_like(xval)
        for i in range(len(xmaxvals)):
            xmax[cumlens[i]:cumlens[i + 1]] = xmaxvals[i]

    if len(xmax) != n:
        raise RuntimeError(f"Length of the xmax vector ({len(xmax)}) should be equal to # design variables ({n})")

    dx = xmax - xmin
    offset = asyinit * np.ones(n) if adaptive_movelimit else 1.0
    xold1, xold2 = None, None

    # Move limit
    if hasattr(move, "__len__"):
        # Set movelimit in case of multiple are given
        move_input = np.asarray(move).copy()
        if move_input.size == len(variables):
            move = np.zeros_like(xval)
            for i in range(move_input.size):
                move[cumlens[i]:cumlens[i + 1]] = move_input[i]
        elif len(move) != n:
            raise RuntimeError(f"""Length of the move vector ({len(move)}) should be equal to number of 
                                design variable signals ({len(variables)}) or total number of 
                                design variables ({n}).""")

    # Get function
    if function is None:
        function = Network.active[0]

    fcur = 0.0
    for iter in range(maxit):
        # Reset all signals in function block
        function.reset()

        # Set the new states
        for i, s in enumerate(variables):
            if cumlens[i + 1] - cumlens[i] == 1:
                s.state = xval[cumlens[i]]
            else:
                s.state = xval[cumlens[i]:cumlens[i + 1]]

        if iter > 0 or any([s.state is None for s in responses]):
            # Calculate response; first iteration may already be calculated
            function.response()

        xval, _ = _concatenate_to_array([s.state for s in variables])

        # Save response
        f = ()
        for s in responses:
            if np.size(s.state) != 1:
                raise TypeError("State of responses must be scalar.")
            if np.iscomplexobj(s.state):
                raise TypeError("Responses must be real-valued.")
            f += (s.state,)

        # Check function change convergence criterion
        fprev, fcur = fcur, responses[0].state
        rel_fchange = abs(fcur - fprev) / abs(fcur)
        if rel_fchange < tolf:
            if verbosity >= 1:
                print(f"SLP converged: Relative function change |Δf|/|f| ({rel_fchange}) below tolerance ({tolf})")
            break

        # Calculate and save sensitivities
        df = ()
        for i, s_out in enumerate(responses):
            for s in responses:
                s.reset()

            s_out.sensitivity = s_out.state * 0 + 1.0

            function.sensitivity()

            sens_list = []
            for v in variables:
                sens_list.append(v.sensitivity if v.sensitivity is not None else 0 * v.state)
            dff, _ = _concatenate_to_array(sens_list)
            df += (dff,)

            # Reset sensitivities for the next response
            function.reset()

        if verbosity >= 3:
            # Display info on variables
            show_sensitivities = verbosity >= 4
            msg = ""
            for i, s in enumerate(variables):
                if show_sensitivities:
                    msg += "{0:>10s} = ".format(s.tag[:10])
                else:
                    msg += f"{s.tag} = "

                # Display value range
                fmt = "% .2e"
                minval, maxval = np.min(s.state), np.max(s.state)
                mintag, maxtag = fmt % minval, fmt % maxval
                if mintag == maxtag:
                    if show_sensitivities:
                        msg += f"       {mintag}      "
                    else:
                        msg += f" {mintag}"
                else:
                    sep = "…" if len(s.state) > 2 else ","
                    msg += f"[{mintag}{sep}{maxtag}]"
                    if show_sensitivities:
                        msg += " "

                if show_sensitivities:
                    # Display info on sensivity values
                    for j, s_out in enumerate(responses):
                        msg += "| {0:s}/{1:11s} = ".format("d" + s_out.tag, "d" + s.tag[:10])
                        minval = np.min(df[j][cumlens[i] : cumlens[i + 1]])
                        maxval = np.max(df[j][cumlens[i] : cumlens[i + 1]])
                        mintag, maxtag = fmt % minval, fmt % maxval
                        if mintag == maxtag:
                            msg += f"       {mintag}      "
                        else:
                            sep = "…" if cumlens[i + 1] - cumlens[i] > 2 else ","
                            msg += f"[{mintag}{sep}{maxtag}] "
                    msg += "\n"
                elif i != len(variables) - 1:
                    msg += ", "
            print(msg)

        # # ASYMPTOTES
        # Calculation of the asymptotes low and upp :
        # For iter = 1,2 the asymptotes are fixed depending on asyinit
        if xold1 is not None and xold2 is not None and adaptive_movelimit:
            # depending on if the signs of xval - xold and xold - xold2 are opposite, indicating an oscillation
            # in the variable xi
            # if the signs are equal the asymptotes are slowing down the convergence and should be relaxed

            # check for oscillations in variables
            # if zzz positive no oscillations, if negative --> oscillations
            zzz = (xval - xold1) * (xold1 - xold2)
            # decrease those variables that are oscillating equal to asydecr
            offset[zzz > 0] *= asyincr
            offset[zzz < 0] *= asydecr

            # check with minimum and maximum bounds of asymptotes, as they cannot be to close or far from the variable
            # give boundaries for upper and lower asymptotes
            offset = np.clip(offset, 1 / (asybound**2), 1.0)

        max_dx = offset * move * dx

        lb = np.maximum(xmin, xval-max_dx)
        ub = np.minimum(xmax, xval+max_dx)
        Ac = np.vstack(df[1:]) if len(df) > 1 else None
        bc = Ac @ xval - np.hstack(f[1:]) if len(df) > 1 else None
        
        res = linprog(df[0], Ac, bc, bounds=np.vstack([lb, ub]).T, options={"disp": False})
        if not res.success:
            print(res)
            return
        xnew = res.x

        # Stopping criteria on step size
        rel_stepsize = np.linalg.norm((xval - xnew) / dx) / np.linalg.norm(xval / dx)
        if rel_stepsize < tolx:
            if verbosity >= 1:
                print(f"SLP converged: Relative stepsize |Δx|/|x| ({rel_stepsize}) below tolerance ({tolx})")
            break

        if verbosity >= 2:
            # Display iteration status message
            msgs = ["g{0:d}({1:s}): {2:+.4e}".format(i, s.tag, f[i]) for i, s in enumerate(responses)]
            if len(responses) > 1:
                max_infeasibility = max(f[1:len(responses)])
                is_feasible = max_infeasibility <= 0
                feasibility_tag = "[f] " if is_feasible else "[ ] "
            else:
                feasibility_tag = ""  # No constraints, so always feasible :)

            print("It. {0: 4d}, {1:s}{2}".format(iter, feasibility_tag, ", ".join(msgs)))

        if verbosity >= 3:
            # Report design feasibility
            g_orig = f[:len(responses)]
            if g_orig[1:].size > 0:
                iconst_max = np.argmax(g_orig[1:])
                print(
                    f"  | {np.sum(g_orig[1:] > 0)} / {len(g_orig) - 1} violated constraints, "
                    f"max. violation ({responses[iconst_max + 1].tag}) = {'%.2g' % g_orig[iconst_max + 1]}"
                )

            # Print design changes
            change_msgs = []
            for i, s in enumerate(variables):
                minchg = np.min(
                    abs(xval[cumlens[i] : cumlens[i + 1]] - xnew[cumlens[i] : cumlens[i + 1]])
                )
                maxchg = np.max(
                    abs(xval[cumlens[i] : cumlens[i + 1]] - xnew[cumlens[i] : cumlens[i + 1]])
                )
                fmt = "%.2g"
                mintag, maxtag = fmt % minchg, fmt % maxchg

                if mintag == maxtag:
                    change_msgs.append(f"Δ({s.tag}) = {mintag}")
                else:
                    change_msgs.append(f"Δ({s.tag}) = {mintag}…{maxtag}")

            print(f"  | Changes: {', '.join(change_msgs)}")

        # Update design vector
        xold2, xold1, xval = xold1, xval, xnew
