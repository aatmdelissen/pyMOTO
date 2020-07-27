import numpy as np


def finite_difference(blk, dx: float = 1e-8, tol: float = 1e-5, random: bool = True):
    """ Performs a finite difference check on the given module or interconnection

    :param blk: The module or interconnection
    :param dx: Perturbation size
    :param random: Randomize sensitivity data
    :param tol: Tolerance
    """
    print("=========================================================================================================\n")
    print("Starting a finite difference check of \"{0}\" with dx = {1}, and tol = {2}".format(
        type(blk).__name__, dx, tol))
    print("Inputs:")
    [print("{}\t{} = {}".format(i, s.tag, s.get_state())) for i, s in enumerate(blk.sig_in)]
    print("")

    # Setup some internal storage
    f0 = [None for _ in blk.sig_out]  # Response values at original inputs
    df_an = [np.empty(0) for _ in blk.sig_out]  # Analytical output sensitivities
    dx_an = [[np.empty(0) for _ in blk.sig_in] for _ in blk.sig_out]

    # Initial reset in case some memory is still left
    blk.reset()

    # Perform response
    blk.response()

    print("Outputs:")
    # Get analytical response and sensitivities, by looping over all outputs
    for Iout, Sout in enumerate(blk.sig_out):
        # Obtain the output state
        output = Sout.get_state()

        print("{}\t{} = {}".format(Iout, Sout.tag, output))

        # Store the output value
        f0[Iout] = (output.copy() if hasattr(output, "copy") else output)

        # Get the output state shape
        shape = (output.shape if hasattr(output, "shape") else ())

        # Generate a (random) sensitivity for output signal
        if random:
            df_an[Iout] = np.random.rand(*shape)
        else:
            df_an[Iout] = np.ones(*shape)

        if np.iscomplexobj(output):
            if random:
                df_an[Iout] = df_an[Iout] + 1j * np.random.rand(*shape)
            else:
                df_an[Iout] = df_an[Iout] + 1j * np.ones(*shape)

        # Set the output sensitivity
        Sout.set_sens(df_an[Iout])

        # Perform the analytical sensitivity calculation
        blk.sensitivity()

        # Store all input sensitivities for this output
        for Iin, Sin in enumerate(blk.sig_in):
            sens = Sin.get_sens()
            dx_an[Iout][Iin] = (sens.copy() if hasattr(sens, "copy") else sens)

        # Reset the sensitivities for next output
        blk.reset()

    # Perturb each of the input signals
    for Iin, Sin in enumerate(blk.sig_in):
        print("___________________________________________________")
        print("Perturbing input {} \"{}\"...\n".format(Iin, Sin.tag))

        # Get input state
        x = Sin.get_state()

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
            else:
                x0 = float(it[0])
                Sin.set_state(x0 + dx)

            # Calculate perturbed solution
            blk.response()

            # Obtain all perturbed responses
            for Iout, Sout in enumerate(blk.sig_out):
                # Obtain perturbed response
                fp = Sout.get_state()

                # Finite difference sensitivity
                df = (fp - f0[Iout])/dx
                dgdx_fd = np.real(np.sum(df*np.conj(df_an[Iout])))

                if dx_an[Iout][Iin] is not None:
                    dgdx_an = (np.real(dx_an[Iout][Iin][it.multi_index] if is_iterable else dx_an[Iout][Iin]))
                else:
                    dgdx_an = 0.0

                if abs(dgdx_an) < tol:
                    error = abs(dgdx_fd - dgdx_an)
                else:
                    error = abs(dgdx_fd - dgdx_an)/max(abs(dgdx_fd), abs(dgdx_an))

                print("d%s/d%s     i = %s\tAn :% .3e\tFD : % .3e\tError: % .3e %s"
                      % (Sout.tag, Sin.tag, it.multi_index, dgdx_an, dgdx_fd, error, "<--*" if error > tol else ""))

            # Restore original state
            if is_iterable:
                it[0] = x0
            else:
                Sin.set_state(x0)

            # If the input state is complex, also do a complex perturbation
            if np.iscomplexobj(x0):
                # Do the perturbation
                if is_iterable:
                    it[0] += dx*1j
                else:
                    print("is x (= {}) equal to x0 (= {})???".format(x, x0))
                    Sin.set_state(x0 + dx*1j)

                # Calculate perturbed solution
                blk.response()

                # Obtain all perturbed responses
                for Iout, Sout in enumerate(blk.sig_out):
                    # Obtain perturbed response
                    fp = Sout.get_state()

                    # Finite difference sensitivity
                    df = (fp - f0[Iout])/dx
                    dgdx_fd = np.real(np.sum(df*np.conj(df_an[Iout])))

                    if dx_an[Iout][Iin] is not None:
                        dgdx_an = (np.imag(dx_an[Iout][Iin][it.multi_index] if is_iterable else dx_an[Iout][Iin]))
                    else:
                        dgdx_an = 0.0

                    if abs(dgdx_an) < tol:
                        error = abs(dgdx_fd - dgdx_an)
                    else:
                        error = abs(dgdx_fd - dgdx_an)/max(abs(dgdx_fd), abs(dgdx_an))

                    print("d%s/d%s (I) i = %s\tAn :% .3e\tFD : % .3e\tError: % .3e %s"
                          % (Sout.tag, Sin.tag, it.multi_index, dgdx_an, dgdx_fd, error, "<--*" if error > tol else ""))

                # Restore original state
                if is_iterable:
                    it[0] = x0
                else:
                    Sin.set_state(x0)

            # Go to the next entry in the array
            it.iternext()

    print("___________________________________________________")
    print("\nFINISHED FINITE DIFFERENCE OF \"{}\"\n".format(type(blk).__name__))
    print("=========================================================================================================\n")
