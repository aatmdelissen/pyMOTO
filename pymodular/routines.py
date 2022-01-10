import numpy as np
from .utils import _parse_to_list, _concatenate_to_array
from .core_objects import Signal, Module, Network
from typing import List, Iterable, Union, Callable


def finite_difference(blk: Module, fromsig: Union[Signal, Iterable[Signal]] = None,
                      tosig: Union[Signal, Iterable[Signal]] = None,
                      dx: float = 1e-8, tol: float = 1e-5, random: bool = True, use_df: list = None,
                      test_fn: Callable = None):
    """ Performs a finite difference check on the given module or network

    :param blk: The module or network
    :param fromsig: Specify input signals of interest
    :param tosig: Specify output signals of interest
    :param dx: Perturbation size
    :param tol: Tolerance
    :param random: Randomize sensitivity data
    :param use_df: Give pre-defined sensitivity data
    :param test_fn: A generic test function (x, dx, df_an, df_fd)
    """
    print("=========================================================================================================\n")
    print("Starting finite difference of \"{0}\" with dx = {1}, and tol = {2}".format(type(blk).__name__, dx, tol))

    if fromsig is None:
        inps = blk.sig_in
    else:
        inps = _parse_to_list(fromsig)

    if tosig is None:
        outps = blk.sig_out
    else:
        outps = _parse_to_list(tosig)

    if isinstance(blk, Network):
        i_first, i_last = -1, -1
        for i, b in enumerate(blk.mods):
            # Find the first module which requires any of the input signals
            if i_first<0 and any([s in inps for s in b.sig_in]):
                i_first = i
            # Find the last module that generates any of the output signals
            if any([s in outps for s in b.sig_out]):
                i_last = i
        blks_pre = Network(blk.mods[:i_first])
        blk = Network(blk.mods[i_first:i_last+1])
        # blks_post = Network(blk.mods[i_last+1:])
        blks_pre.response()

    print("Inputs:")
    [print("{}\t{} = {}".format(i, s.tag, s.state)) for i, s in enumerate(inps)]
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
    # Get analytical response and sensitivities, by looping over all outputs
    for Iout, Sout in enumerate(outps):
        # Obtain the output state
        output = Sout.state

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

        while not it.finished:
            # Get original value and do the perturbation
            if is_iterable:
                x0 = it[0].copy()
                it[0] += dx
                Sin.state = x
            else:
                x0 = np.asscalar(it[0])
                Sin.state = x0 + dx

            # Calculate perturbed solution
            blk.response()

            # Obtain all perturbed responses
            for Iout, Sout in enumerate(outps):
                # Obtain perturbed response
                fp = Sout.state
                # print(fp)

                # Finite difference sensitivity
                df = (fp - f0[Iout])/dx
                dgdx_fd = np.real(np.sum(df*np.conj(df_an[Iout])))

                if dx_an[Iout][Iin] is not None:
                    try:
                        dgdx_an = np.real(dx_an[Iout][Iin][it.multi_index])
                    except IndexError:
                        dgdx_an = np.real(dx_an[Iout][Iin])
                else:
                    dgdx_an = 0.0

                if abs(dgdx_an) < tol:
                    error = abs(dgdx_fd - dgdx_an)
                else:
                    error = abs(dgdx_fd - dgdx_an)/max(abs(dgdx_fd), abs(dgdx_an))

                print("d%s/d%s     i = %s\tAn :% .3e\tFD : % .3e\tError: % .3e %s"
                      % (Sout.tag, Sin.tag, it.multi_index, dgdx_an, dgdx_fd, error, "<--*" if error > tol else ""))
                if error > tol:
                    print("ERROR!")

                if test_fn is not None:
                    test_fn(x0, dx, dgdx_an, dgdx_fd)

            # Restore original state
            if is_iterable:
                it[0] = x0
                Sin.state = x
            else:
                Sin.state = x0

            # If the input state is complex, also do a complex perturbation
            if np.iscomplexobj(x0):
                # Do the perturbation
                if is_iterable:
                    it[0] += dx*1j
                else:
                    Sin.state = x0 + dx*1j

                # Calculate perturbed solution
                blk.response()

                # Obtain all perturbed responses
                for Iout, Sout in enumerate(outps):
                    # Obtain perturbed response
                    fp = Sout.state

                    # Finite difference sensitivity
                    df = (fp - f0[Iout])/dx
                    dgdx_fd = np.real(np.sum(df*np.conj(df_an[Iout])))

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

                    print("d%s/d%s (I) i = %s\tAn :% .3e\tFD : % .3e\tError: % .3e %s"
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

    print("___________________________________________________")
    print("\nFINISHED FINITE DIFFERENCE OF \"{}\"\n".format(type(blk).__name__))
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


def minimize_mma(function, variables, responses, tolx=1e-4, tolf=1e-4, maxit=100, xmin=0.0, xmax=1.0):
    # Save initial state
    xval, cumlens = _concatenate_to_array([s.state for s in variables])
    n = len(xval)

    def fi(_, grad, i):

        if grad.size > 0:
            # Calculate sensitivities
            function.reset()

            responses[i].sensitivity = 1.0

            function.sensitivity()

            grad[:], _ = _concatenate_to_array(obtain_sensitivities(variables))

        return responses[i].state

    global it
    it = 0

    # Objective function
    def f0(x, grad):
        global it
        # Set the new states
        for i, s in enumerate(variables):
            s.state = x[cumlens[i]:cumlens[i + 1]]

        # Calculate response
        function.response()

        print("It. {0: 4d}, f0 = {1: .2e}, f1 = {2: .2e}".format(it, responses[0].state, responses[1].state))
        it += 1

        return fi(x, grad, 0)

    # Create optimization
    import nlopt

    opt = nlopt.opt(nlopt.LD_MMA, n)

    opt.set_min_objective(f0)
    for ri in range(1, len(responses)):
        opt.add_inequality_constraint(lambda x, grad: fi(x, grad, ri))

    opt.set_lower_bounds(np.ones_like(xval) * xmin)
    opt.set_upper_bounds(np.ones_like(xval) * xmax)

    opt.set_maxeval(maxit)
    opt.set_xtol_rel(tolx)
    opt.set_ftol_rel(tolf)

    opt.optimize(xval)


