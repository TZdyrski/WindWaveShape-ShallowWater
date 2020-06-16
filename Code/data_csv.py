#!/usr/bin/env python3
# data_csv.py
import numpy as np
import pandas as pd
import glob
import re

def save_data(data, suffix, stack_coords=False, **kwargs):

    extension = '.csv'

    # Only include a subset of parameters which could feasibly appear in
    # filename naming scheme
    filename_param_subset = ['eps', 'mu', 'P', 'forcing_type', 'wave_type']
    filename_params = {k:v for k,v in kwargs.items() if k in
            filename_param_subset}

    filename = suffix +\
            ''.join('_{}{}'.format(
                str(k),
                str(round(v,3) if isinstance(v,float) else v)
                ) for k,v in
                sorted(filename_params.items()))+\
            extension

    parameter_comment = '#'+\
            ''.join(' {}={}'.format(str(k),str(v)) for k,v in sorted(kwargs.items()))

    with open(filename, 'w') as f:
        # Write comment to top of file
        f.write(parameter_comment+'\n')

        if stack_coords:
            # Convert column index to a row so that row index is saved in csv
            # output.
            # Source: https://stackoverflow.com/questions/35047842
            data.to_pandas().T.reset_index().T.to_csv(f)
        else:
            data.to_dataframe().to_csv(f)

    return

def load_data(filename, extension='.csv', stack_coords=True):

    if stack_coords:
        # Skip 2nd row, which contains column headers
        data = pd.read_csv(filename, index_col=0, comment='#',
                skiprows=[2])

        # Read column headers separately
        column_header = pd.read_csv(filename, index_col=0, skiprows=2,
                nrows=1, header=None)

        # Save column headers
        data.columns = column_header.values.flatten()
        data.columns.name = column_header.index[0]

        # Restack DataFrame to convert to xarray
        data = data.stack()
    else:
        data = pd.read_csv(filename, index_col=0, comment='#')

    data = data.to_xarray()

    # Read in first line of file (which is commented out by #, so it is
    # skipped by read_csv) giving the parameters
    with open(filename, 'r') as f:
        parameter_comment = f.readline()

    # Remove comment symbol # (first character) and newline (last character)
    parameter_comment = parameter_comment[1:-1]

    # Convert to dict
    parameters = {pair[0] : pair[1] for pair
            in re.findall("\s([\w\.\-\*\(\)]+)=([\w\.\-\*\(\)]+)", parameter_comment)}

    # Convert all values to floats and replace all inf with np.inf
    def conv_to_float(string):
        try:
            return float(string)
        except ValueError:
            return string

    parameters = {k : (conv_to_float(v) if v != 'inf' else 'np.inf') for k,v in
            parameters.items()}

    # Add parameters to attributes field of data
    data.attrs = parameters

    return data

def find_filenames(load_prefix, filename_base, required_words=[],
        parameters=dict(), allow_multiple_files=False):

    if parameters.get('wave_type', None) == 'solitary':
        # Cannot specify both 'wave_type'='solitary' and 'mu'
        parameters.pop('mu')

    filenames = glob.glob(load_prefix+filename_base+'*')

    if len(filenames) < 1:
        raise(OSError("'"+filename_base+"' file not found in "+load_prefix))

    # Only include a subset of parameters which could feasibly appear in
    # filename naming scheme
    filename_params = ['eps', 'mu', 'P', 'forcing_type', 'wave_type']
    parameters = {k:v for k,v in parameters.items() if k in
            filename_params}

    # Add required_words to parameters dict as keys without values
    parameters = {**{k : '' for k in required_words}, **parameters}

    for param in parameters:
        # Select filenames with correct param
        filenames = [filename for filename in filenames if
                param+str(parameters[param]) in filename]

    if len(filenames) < 1:
        # There should only be one option left
        raise(ValueError('No filenames match parameters specifications'))
    elif len(filenames) > 1 and not allow_multiple_files:
        # There should only be one option left
        raise(ValueError('Too many filename choices; narrow down more;'+
            ' remaining filenames: '+str(filenames)))

    if allow_multiple_files:
        return filenames
    else:
        return filenames[0]
