import numpy as np

def round_sig_figs(number, num_sig_figs=3):

    # Round to num_sig_figs significant figures
    # Source:
    # https://stackoverflow.com/questions/3410976/
    round_single_num = lambda number, num_sig_figs : \
            round(number, num_sig_figs-1-
            int(np.floor(np.log10(np.absolute(number)))))

    result = np.vectorize(round_single_num)(number, num_sig_figs)

    # If original input was a scalar, return a scalar
    if np.isscalar(number):
        result = result.item()

    return result
