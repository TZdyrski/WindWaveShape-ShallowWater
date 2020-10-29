import numpy as np
from scipy.fftpack import diff as psdiff

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
        accuracy=4,
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
    accuracy : {2,4}
        Order of accuracy for 'periodic_fd'. Default is 4.
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
        pad_width = [tuple(x) for x in np.zeros((u.ndim,2),int)]
        N = u.shape[axis]
        if accuracy == 2:
            # 2nd-order center difference with periodic boundary conditions
            if order == 1:
                # Pad once on axis
                pad_width[axis] = (1,1)
                N += 2
                derivative = (
                        np.eye(N,k=-1)
                        -np.eye(N,k=1)
                        )/\
                        (2*dx)
            elif order == 2:
                # Pad once on axis
                pad_width[axis] = (1,1)
                N += 2
                derivative = (
                        np.eye(N,k=-1)
                        - 2*np.eye(N,k=0)
                        + np.eye(N,k=1)
                        )/(dx**2)
            elif order == 3:
                # Pad twice on axis
                pad_width[axis] = (2,2)
                N += 4
                derivative = (
                        np.eye(N,k=-2)
                        - 2*np.eye(N,k=-1)
                        + 2*np.eye(N,k=1)
                        - np.eye(N,k=2)
                        )/(2*dx**3)
            elif order == 4:
                # Pad twice on axis
                pad_width[axis] = (2,2)
                N += 4
                derivative = (
                        np.eye(N,k=-2)
                        - 4*np.eye(N,k=-1)
                        + 6*np.eye(N,k=0)
                        - 4*np.eye(N,k=1)
                        + np.eye(N,k=2)
                        )/(dx**4)
            elif order == 5:
                # Pad thrice on axis
                pad_width[axis] = (3,3)
                N += 6
                derivative = (
                        np.eye(N,k=-3)
                        - 4*np.eye(N,k=-2)
                        + 5*np.eye(N,k=-1)
                        - 5*np.eye(N,k=1)
                        + 4*np.eye(N,k=2)
                        - np.eye(N,k=3)
                        )/(2*dx**5)
            elif order == 6:
                # Pad thrice on axis
                pad_width[axis] = (3,3)
                N += 6
                derivative = (
                        + 1*np.eye(N,k=-3)
                        - 6*np.eye(N,k=-2)
                        + 15*np.eye(N,k=-1)
                        - 20*np.eye(N,k=0)
                        + 15*np.eye(N,k=1)
                        - 6*np.eye(N,k=2)
                        + 1*np.eye(N,k=3)
                        )/(dx**6)
            else:
                raise(ValueError("Derivatives of type 'periodic_fd'"+\
                        "are only supported up to order 6, but "+\
                        str(order)+" was given"))
        elif accuracy == 4:
            # 4th-order center difference with periodic boundary conditions
            if order == 1:
                # Pad twice on axis
                pad_width[axis] = (2,2)
                N += 4
                derivative = (
                        - 1/12*np.eye(N,k=-2)
                        + 2/3*np.eye(N,k=-1)
                        - 2/3*np.eye(N,k=1)
                        + 1/12*np.eye(N,k=2)
                        )/\
                        (dx)
            elif order == 2:
                # Pad twice on axis
                pad_width[axis] = (2,2)
                N += 4
                derivative = (
                        - 1/12*np.eye(N,k=-2)
                        + 4/3*np.eye(N,k=-1)
                        - 5/2*np.eye(N,k=0)
                        + 4/3*np.eye(N,k=1)
                        - 1/12*np.eye(N,k=2)
                        )/\
                        (dx**2)
            elif order == 3:
                # Pad thrice on axis
                pad_width[axis] = (3,3)
                N += 6
                derivative = (
                        - 1/8*np.eye(N,k=-3)
                        + 1*np.eye(N,k=-2)
                        - 13/8*np.eye(N,k=-1)
                        + 13/8*np.eye(N,k=1)
                        - 1*np.eye(N,k=2)
                        + 1/8*np.eye(N,k=3)
                        )/(dx**3)
            elif order == 4:
                # Pad thrice on axis
                pad_width[axis] = (3,3)
                N += 6
                derivative = (
                        - 1/6*np.eye(N,k=-3)
                        + 2*np.eye(N,k=-2)
                        - 13/2*np.eye(N,k=-1)
                        + 28/3*np.eye(N,k=0)
                        - 13/2*np.eye(N,k=1)
                        + 2*np.eye(N,k=2)
                        - 1/6*np.eye(N,k=3)
                        )/(dx**4)
            elif order == 5:
                # Pad quatrice on axis
                pad_width[axis] = (4,4)
                N += 8
                derivative = (
                        - 1/6*np.eye(N,k=-4)
                        + 3/2*np.eye(N,k=-3)
                        - 13/3*np.eye(N,k=-2)
                        + 29/6*np.eye(N,k=-1)
                        - 29/6*np.eye(N,k=1)
                        + 13/3*np.eye(N,k=2)
                        - 3/2*np.eye(N,k=3)
                        + 1/6*np.eye(N,k=4)
                        )/(dx**5)
            elif order == 6:
                # Pad quatrice on axis
                pad_width[axis] = (4,4)
                N += 8
                derivative = (
                        - 1/4*np.eye(N,k=-4)
                        + 3*np.eye(N,k=-3)
                        - 13*np.eye(N,k=-2)
                        + 29*np.eye(N,k=-1)
                        - 75/2*np.eye(N,k=0)
                        + 29*np.eye(N,k=1)
                        - 13*np.eye(N,k=2)
                        + 3*np.eye(N,k=3)
                        - 1/4*np.eye(N,k=4)
                        )/(dx**6)
            else:
                raise(ValueError("Derivatives of type 'periodic_fd'"+\
                        "are only supported up to order 6, but "+\
                        str(order)+" was given"))
        else:
            raise(ValueError("Derivatives of type 'periodic_fd'"+\
                    "are only supported for accuracies 2 and 4, but "+\
                    str(accuracy)+" was given"))
        u_padded = np.pad(u, pad_width, 'wrap')
        derivative = np.tensordot(derivative, u_padded,
                axes=([0,axis]))

        # Trim off padding
        trim_num = pad_width[axis][0]
        trim_mat = np.eye(N,N-2*trim_num,k=-trim_num)
        derivative = np.tensordot(trim_mat, derivative,
                axes=([0,axis]))
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
