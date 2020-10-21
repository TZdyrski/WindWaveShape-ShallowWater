import numpy as np

def round_sig_figs(number, num_sig_figs=3):

    # Round to num_sig_figs significant figures
    # Source:
    # https://stackoverflow.com/questions/3410976/
    def round_single_num(number, num_sig_figs):
        if number == 0:
            return number
        else:
            result = round(number, num_sig_figs-1-
                    int(np.floor(np.log10(np.absolute(number)))))
        return result

    result = np.vectorize(round_single_num)(number, num_sig_figs)

    # If original input was a scalar, return a scalar
    if np.isscalar(number):
        result = result.item()

    return result

def derivative(u, dx=1, period=2*np.pi, axis=0, order=1,
        deriv_type='gradient', *args, **kwargs):
    """Calculate the derivative of order 'order'.

    Parameters
    ----------
    u : ndarray
        Data of which to take the derivative.
    dx : float
        Spacing between points in u. Default is 1.
    period : float
        Period of u used for 'FFT' type calculation. Default is 2*np.pi.
    axis : integer
        Axis along which to take the derivative. Default is 0.
    order : integer
        Order of derivative. Default is 1.
    deriv_type : 'gradient', 'FFT', or 'periodic_fd'
        The type of derivative to take. 'gradient' yields a
        non-periodic, finite-difference derivative. 'FFT' yields a
        periodic derivative using the FFT. 'periodic_fd" yields a
        periodic, finite-difference derivative. Default is 'gradient'.

    """
    if deriv_type == 'gradient':
        # Compute the x derivatives using the finite-difference method
        derivative = u
        for n in range(order):
            # Apply derivative 'order' times
            derivative = np.gradient(derivative, dx, axis=axis,
                    edge_order=2)
    elif deriv_type == 'FFT':
        # Compute the x derivatives using the pseudo-spectral method
        derivative = psdiff(u, period=period, order=order)
    elif deriv_type == 'periodic_fd':
        # 2nd-order center difference with periodic boundary conditions
        if order == 1:
            u_padded_once = np.concatenate((u[[-1]], u, u[[0]]))
            derivative = (
                    u_padded_once[2:]
                    -u_padded_once[0:-2]
                    )/\
                    (2*dx)
        elif order == 2:
            u_padded_once = np.concatenate((u[[-1]], u, u[[0]]))
            derivative = (
                    u_padded_once[2:]
                    - 2*u_padded_once[1:-1]
                    + u_padded_once[0:-2]
                    )/(dx**2)
        elif order == 3:
            u_padded_twice = np.concatenate((u[-2:], u, u[0:2]))
            derivative = (
                    u_padded_twice[4:]
                    - 2*u_padded_twice[3:-1]
                    + 2*u_padded_twice[1:-3]
                    - u_padded_twice[0:-4]
                    )/(2*dx**3)
        elif order == 4:
            u_padded_twice = np.concatenate((u[-2:], u, u[0:2]))
            derivative = (
                    u_padded_twice[4:]
                    - 4*u_padded_twice[3:-1]
                    + 6*u_padded_twice[2:-2]
                    - 4*u_padded_twice[1:-3]
                    + u_padded_twice[0:-4]
                    )/(dx**4)
        elif order == 5:
            u_padded_thrice = np.concatenate((u[-3:], u, u[0:3]))
            derivative = (
                    u_padded_thrice[6:]
                    - 4*u_padded_thrice[5:-1]
                    + 5*u_padded_thrice[4:-2]
                    - 5*u_padded_thrice[2:-4]
                    + 4*u_padded_thrice[1:-5]
                    - u_padded_thrice[0:-6]
                    )/(2*dx**5)
        elif order == 6:
            # Calculate third-order accurate derivative (since there is
            # no 2nd order accurate one)
            u_padded_quatrice = np.concatenate((u[-4:], u, u[0:4]))
            derivative = (
                    - 1/4*u_padded_quatrice[8:]
                    + 3*u_padded_quatrice[7:-1]
                    - 13*u_padded_quatrice[6:-2]
                    + 29*u_padded_quatrice[5:-3]
                    - 75/2*u_padded_quatrice[4:-4]
                    + 29*u_padded_quatrice[3:-5]
                    - 13*u_padded_quatrice[2:-6]
                    + 3*u_padded_quatrice[1:-7]
                    - 1/4*u_padded_quatrice[0:-8]
                    )/(dx**6)
        else:
            raise(ValueError("Derivatives of type 'periodic_fd'"+\
                    "are only supported up to order 6, but "+\
                    str(order)+" was given"))
    else:
        raise(ValueError("'deriv_type' must be either 'gradient',"+\
        "'FFT', or 'periodic_fd', but "+deriv_type+" was given"))

    return derivative

def get_var_stats(profile, var='x/h',periodic=True):
    varNum = profile[var].size
    varLen = float(profile[var].max()-profile[var].min())
    dvar = varLen/(varNum-1)
    if periodic:
        # If the domain in 'var' is assumed periodic, then the last
        # point is not included. That is, only
        # 0, dvar, 2*dvar, ..., (varNum-1)*dvar
        # are provided, but the domain is assumed to have length
        # varNum*dvar. Therefore adjust varLen
        varLen = varNum*dvar

    return varLen, varNum, dvar
