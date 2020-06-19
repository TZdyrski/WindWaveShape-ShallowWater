#!/usr/bin/env python3
# postproccesor.py
import sys
import shutil
import glob
import scipy as sp
import scipy.integrate
import scipy.signal
import numpy as np
import xarray as xr
import data_csv

def get_var_stats(profile, var='x/h'):
    varNum = profile[var].size
    varLen = float(profile[var][-1]-profile[var][0])
    dvar = varLen/(varNum-1)

    return varLen, varNum, dvar

def skewness(profile):
    """Get the skewness of the solution calculated with
    solve_system.

    Returns
    -------
    skewness : ndarray(t.size,)
        Returns the skewness calculated at each time step.
    """

    xLen,xNum,dx = get_var_stats(profile)

    # Cast to type double (float64) since fft (used in Hilbert for
    # asymmetry) is unable to hand long double (float128)
    profile = np.array(profile, dtype=np.float64)

    average_sol_cubed = sp.integrate.trapz(
            profile**3,
            dx=dx,
            axis=0)/(xLen)
    average_sol_squared = sp.integrate.trapz(
            profile**2,
            dx=dx,
            axis=0)/(xLen)

    skewness = average_sol_cubed/average_sol_squared**(3/2)

    return skewness

def asymmetry(profile):
    """Get the asymmetry of the solution calculated with
    solve_system.

    Returns
    -------
    asymmetry : ndarray(t.size,)
        Returns the asymmetry calculated at each time step.
    """

    xLen,xNum,dx = get_var_stats(profile)

    # Cast to type double (float64) since fft (used in Hilbert for
    # asymmetry) is unable to hand long double (float128)
    profile = np.array(profile, dtype=np.float64)

    # Calculate the hilbert transform
    # Confusingly, sp.signal.hilbert gives the analytic signal, x <- x + i*H(x)
    profile_hilbert = (sp.signal.hilbert(profile,axis=0) - profile)/1j

    # Throw out imaginary part since it should be zero (to within rounding error)
    profile_hilbert = np.real(profile_hilbert)

    average_hilbert_cubed = sp.integrate.trapz(
            profile_hilbert**3,
            dx=dx,
            axis=0)/(xLen)
    average_sol_squared = sp.integrate.trapz(
            profile**2,
            dx=dx,
            axis=0)/(xLen)

    asymmetry = average_hilbert_cubed/average_sol_squared**(3/2)

    return asymmetry

def maximum(profile):
    """Get the height of the function at each time step.

    Returns
    -------
    height : ndarray(t.size,)
        Returns the maximum calculated at each time step.
    """

    maximum = np.amax(profile, axis=0) - np.amin(profile, axis=0)

    return maximum

def energy(profile):
    _,_,dx = get_var_stats(profile)

    # Calculate energy
    # Convert from eta'^2 = eta^2/h^2 to energy density (energy per unit
    # length) dE' = d(E/rho_w/g/h^2) = rho_w*g*eta^2*dx/rho_w/g/h^2 =
    # (eta/h)^2*dx = (eta')^2*dx
    # (Primes denote the nondim variables used throughout this solver)
    energy_density = profile**2*dx

    # Convert from energy density dE' = d(E/h^2/rho_w/g) to energy
    # E' = E/rho_w/g/h^2 = int dE' = int (eta')^2*dx
    # Note: this is the total energy contained in the domain; ie, if
    # there are n wavelengths, this is n*(energy per wavelength)
    energy = np.sum(energy_density,axis=0)

    return energy

def spatial_fourier_transform(profile, repeat_times = 5):
    xLen,xNum,dx = get_var_stats(profile)

    # Resample (via interpolation) to get a higher FFT resolution
    profileRepeated = np.tile(profile, (repeat_times,1))

    # Take spatial FFT \hat'{eps*eta'} = \hat{eps*eta'}*k_E = \hat{eta}*k_E/h
    # (Primes denote the nondim variables used throughout this solver)
    # (scale FFT by 1/N so the FFT of a sinusoid has unit amplitude;
    # this also gives it the same units as the continuous Fourier
    # Transform)
    profileFFT = np.fft.fft(profileRepeated,
            axis=0)/xNum/repeat_times

    # Generate spatial FFT conjugate coordinate
    # Convert from matplotlib's wavenumber in cycles per x-unit to our
    # radians per x-unit by multiplying by 2*pi radians/cycle
    kappa = np.fft.fftfreq(xNum*repeat_times, dx)*2*np.pi
    # Convert from kappa' = kappa/k_E to
    #  kappa'*sqrt(mu) = kappa/k_E*h*k_E = kappa*h
    kappa = kappa*np.sqrt(float(profile.attrs['mu']))

    # Shift zero to center
    kappa = np.fft.fftshift(kappa)
    profileFFT = np.fft.fftshift(profileFFT,axes=0)

    # Repackage as a DataArray
    fourier_timeseries = xr.DataArray(profileFFT,
            dims=('kappa*h', 't*eps*sqrt(g*h)*k_E'),
            coords={'kappa*h' : kappa, 't*eps*sqrt(g*h)*k_E' :
                profile['t*eps*sqrt(g*h)*k_E']}, attrs=profile.attrs)

    return fourier_timeseries

def convert_k_E_to_k(signal):
    # Convert from kappa'*h' = kappa/k_E*h*k_E = kappa*h to
    # kappa'*h'/sqrt(mu)/k' = kappa*h/(k_E*h)/k*k_E = kappa/k
    signal = signal.assign_coords({'kappa/k' :
        signal['kappa*h']/float(signal.attrs['mu'])/\
                (2*np.pi/float(signal.attrs['wave_length']))})

    # Replace kappa*h with kappa/k as dimensional coordinate
    signal = signal.swap_dims({'kappa*h' : 'kappa/k'})

    # Remove kappa/h coordinate
    signal = signal.drop('kappa*h')

    return signal

def power_spectrum(fourier_transformed):
    # Calculate power spectrum (ie abs squared) \abs{\hat'{eps*eta'}}^2 =
    # \abs{\hat{eta}}^2*k_E^2/h^2
    # (Primes denote the nondim variables used throughout this solver)
    power_spec = np.absolute(fourier_transformed)**2
    power_spec.attrs = fourier_transformed.attrs

    return power_spec

def temporal_fourier_transform(signal, rel_tol=1e-3):

    tLen, tNum, dt = get_var_stats(signal, var='t*eps*sqrt(g*h)*k_E')

    # Convert from matplotlib's wavenumber in cycles per t1-unit to our
    # radians per t-unit by multiplying by 2*pi radians/cycle
    omega = np.fft.fftfreq(tNum, dt)*2*np.pi
    # Convert from omega' = omega/k_E/sqrt(g*h)/eps to
    #  omega'/k' = omega/k_E/sqrt(g*h)/k*k_E/eps = omega/sqrt(g*h)/k*eps
    # Note: the factor of eps comes from the fact that this is the
    # t1=t*eps frequency
    omega = omega*float(signal.attrs['wave_length'])/2/np.pi

    # Shift zero to center
    omega = np.fft.fftshift(omega)

    # Make a mesh of unshifted frequencies first for multiplying each
    # mode by exp(sqrt(1+P)*k*t)
    omega_mesh,kappa_mesh = np.meshgrid(omega,signal['kappa/k'])

    # Set kappa-columns with small values equal to zero so we don't blow them up
    tol = np.amax(np.abs(signal))*rel_tol

    less_than_tol = np.abs(signal)<tol

    kappa_always_small = less_than_tol.all(dim='t*eps*sqrt(g*h)*k_E')

    signal[{'kappa/k' : kappa_always_small}] = 0

    # Multiply each mode by exp(sqrt(1+P)*k*t)
    signal_modified = np.exp(
            -np.imag(1/2*float(signal.attrs['eps'])*1j*float(signal.attrs['P']))
            *2*np.pi/float(signal.attrs['wave_length'])
            *np.abs(kappa_mesh)
            *signal['t*eps*sqrt(g*h)*k_E'].values
            /4.14027 # fudge factor
            )*signal

    # Take temporal FFT \hat'{\hat'{eps*eta'}} =
    # \hat'{\hat{eps*eta'}}*k_E = \hat{\hat{eps*eta'}}*k_E^2*sqrt(g*h) =
    # \hat{\hat{eta}} k_E^2*sqrt(g/h)
    # (Primes denote the nondim variables used throughout this solver)
    # (scale FFT by 1/N so the FFT of a sinusoid has unit amplitude;
    # this also gives it the same units as the continuous Fourier
    # Transform)
    signal_double_fourier = np.fft.fft(signal_modified, axis=1)/tNum

    # Shift zero to center
    signal_double_fourier = np.fft.fftshift(signal_double_fourier,axes=1)

    signal_double_fourier = xr.DataArray(signal_double_fourier,
            dims=('kappa/k', 'omega/sqrt(g*h)/k/eps'),
            coords={'kappa/k' : signal['kappa/k'],
                'omega/sqrt(g*h)/k/eps' : omega}, attrs=signal.attrs)

    return signal_double_fourier

def biphase(profile):
    # Take spatial FFT \hat'{eps*eta'} = \hat{eps*eta'}*k_E = \hat{eta}*k_E/h
    # (Primes denote the nondim variables used throughout this solver)
    # (scale FFT by 1/N so the FFT of a sinusoid has unit amplitude;
    # this also gives it the same units as the continuous Fourier
    # Transform)
    fourier = spatial_fourier_transform(profile)
    power_spec = power_spectrum(fourier)

    # Only include first 2 peaks
    power_spec_peak_indices = find_initial_peaks(power_spec, num_peaks=2)

    # Get indices for primary (m=1) and first harmonic (m=2)
    primaryIndex = power_spec_peak_indices[0]
    firstHarmonicIndex = power_spec_peak_indices[1]

    # Calculate bispectra
    bispectra = fourier[primaryIndex,:]**2*\
            np.conjugate(fourier[firstHarmonicIndex])

    # Calculate biphase
    biphase = np.angle(bispectra)

    return biphase

def find_initial_peaks(signal, num_peaks=5):
    signal_initial = signal[{'t*eps*sqrt(g*h)*k_E':0}].reset_coords(
            't*eps*sqrt(g*h)*k_E', drop=True)
    # Find peaks in initial data (peaks shouldn't move over time either)
    signal_peaks = sp.signal.find_peaks(signal_initial)[0]

    # Determine the name of the coordinate (there should be only one
    # since we removed the time coordinate)
    coordinate = list(signal_initial.coords.keys())[0]

    # Only keep non-negative kappa peaks
    signal_peaks = signal_peaks[
            np.array(
                signal_initial[coordinate][signal_peaks] >= 0
                )
            ]

    # Sort the peaks
    signal_peaks = signal_peaks[
            np.argsort(
                np.array(
                    signal_initial[{coordinate:signal_peaks}]
                    )
                )[::-1]
            ]

    # Only include first num_peaks
    signal_peaks = signal_peaks[0:num_peaks]

    return signal_peaks

def truncate_after_peak(signal, num_peaks=3):
    signal_initial = np.abs(signal[{'t*eps*sqrt(g*h)*k_E':0}])

    # Only include first num_peaks+1 (we need one extra since the
    # right-sided base always gives the right window limit, so we'll use
    # the n+1th left base limit to define the nth right base limit)
    signal_peaks = find_initial_peaks(np.abs(signal), num_peaks=num_peaks+1)

    # Find peak bases in initial data (peaks shouldn't move over time either)
    signal_left_base_indices = sp.signal.peak_prominences(
            signal_initial, signal_peaks)[1]

    # Use the n+1th left base limit to define the nth right base limit
    signal_right_base_indices = signal_left_base_indices[1:]
    signal_left_base_indices= signal_left_base_indices[:-1]

    # Take the last base indices
    last_base_index = signal_right_base_indices[-1]

    # Calculate the last base value
    last_base = signal_initial['kappa/k'][last_base_index].item()

    # Cut off after the last base (add 1 since we want the last base
    # *inclusive)
    signal = signal.sel({'kappa/k' : slice(0, last_base)})

    return signal

def annotate_peak_locations(signal, mode_num=1):
    signal_initial = np.abs(signal[{'t*eps*sqrt(g*h)*k_E':0}])

    # Only include first num_peaks
    signal_peaks = find_initial_peaks(np.abs(signal), num_peaks=mode_num)

    # Store height (specifically, the largest height as a function of
    # time) and bases of second peak for inset image
    signal_peak_height = np.amax(signal[
        {'kappa/k':signal_peaks[mode_num-1]}]).item()

    # Find bases of mode_num-th peak in initial data (peaks shouldn't
    # move over time either) Calculate using 0.9999 prominence since we
    # want it cropped closely to the peak; the truncate_after_peak base
    # finding prescription would go from the left side of mode_num-th
    # peak to the left side of mode_num+1-th peak, including all the
    # intervening flat space.
    signal_peak_base_indices = sp.signal.peak_widths(
            signal_initial, [signal_peaks[mode_num-1]],
            rel_height=0.9999)[2:]

    # Convert from indices to kappa values by multiplying by the
    # kappa[1] (since all steps are evenly spaced)
    _,_,dkappa = get_var_stats(signal_initial, var='kappa/k')
    signal_peak_base = np.array(signal_peak_base_indices).flatten()*round(dkappa,2)

    # Cut off after the last base (add 1 since we want the last base
    # *inclusive)
    peak_info = {
            str(mode_num)+'-mode_left_base' : signal_peak_base[0],
            str(mode_num)+'-mode_right_base' : signal_peak_base[1],
            str(mode_num)+'-mode_height' : signal_peak_height,
            }

    return peak_info

def time_fractions(signal):
    # Trim snapshots
    snapshot_fracs = np.array([0,1/3,2/3,3/3])

    tLen = float(signal['t*eps*sqrt(g*h)*k_E'][-1])
    tNum = signal['t*eps*sqrt(g*h)*k_E'].size

    # Convert snapshot fractions to snapshot times
    snapshot_ts = snapshot_fracs*tLen

    # Convert snapshot times to snapshot indexes
    snapshot_indxs = np.floor(snapshot_ts*tNum/tLen).astype(int)

    # Ensure that indexes do not underflow
    ts_too_big_indxs = snapshot_indxs < 0
    snapshot_indxs[ts_too_big_indxs] = 0

    # Ensure that indexes do not overflow
    ts_too_big_indxs = snapshot_indxs >= tNum
    snapshot_indxs[ts_too_big_indxs] = tNum-1

    signal = signal[{'t*eps*sqrt(g*h)*k_E':snapshot_indxs}]

    return signal

def time_downsample(signal, downsample_factor=1):
    # Down sample the time component to save space since we don't need
    # the higher frequencies
    signal_downsampled = sp.signal.decimate(signal, downsample_factor)
    tResampled = sp.signal.decimate(signal['t*eps*sqrt(g*h)*k_E'],
            downsample_factor)

    signal_downsampled = xr.DataArray(signal_downsampled,
            dims=signal.dims, coords={**signal.coords,
                't*eps*sqrt(g*h)*k_E' : tResampled}, attrs=signal.attrs)

    return signal_downsampled

def generate_statistics(filename):
    # Extract data
    data_array = data_csv.load_data(filename)

    # Calculate max height
    maximums = maximum(data_array)
    # Normalize by initial maximum
    maximums_normalized = maximums/maximums[{'t*eps*sqrt(g*h)*k_E':0}]

    # Calculate skewness
    skewnesses = skewness(data_array)

    # Calculate asymmetry
    asymmetries = asymmetry(data_array)

    save_biphase = data_array.attrs['wave_type'] != 'solitary'
    if save_biphase:
        # Calculate biphase
        biphases = biphase(data_array)

    # Calculate energy
    energies = energy(data_array)
    # Normalize by initial energy
    energies_normalized = energies/energies[{'t*eps*sqrt(g*h)*k_E':0}]

    # Combine shape statistics
    statistics = xr.Dataset(data_vars = {
        'max(eta)/max(eta_0)' : ('t*eps*sqrt(g*h)*k_E' ,
            maximums_normalized),
        'skewness' : ('t*eps*sqrt(g*h)*k_E' , skewnesses),
        'asymmetry' : ('t*eps*sqrt(g*h)*k_E' , asymmetries),
        **({'biphase' : ('t*eps*sqrt(g*h)*k_E' , biphases)} if
            save_biphase else {}),
        'E/E_0' : ('t*eps*sqrt(g*h)*k_E' ,
            energies_normalized),
        },
        coords = {'t*eps*sqrt(g*h)*k_E' :
            data_array['t*eps*sqrt(g*h)*k_E']},
        attrs=data_array.attrs)

    return statistics

def process_shape_statistics(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Create shape statistics
        statistics = generate_statistics(filename)

        # Save statistics
        data_csv.save_data(statistics, save_prefix+'Shape-Statistics',
                **statistics.attrs)

def process_power_spec_vs_kappa(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'cnoidal'},
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename)

        # Calculate power spectrum
        power_spec = power_spectrum(spatial_fourier_transform(
            data_array))

        # Convert from kappa*h to kappa/k coordinate
        power_spec = convert_k_E_to_k(power_spec)

        # Trim times
        power_spec = time_fractions(power_spec)

        # Trim to num_peaks modes
        power_spec = truncate_after_peak(power_spec)

        # Get peak information
        peak_info = annotate_peak_locations(power_spec, mode_num=2)

        # Save power spectrum
        data_csv.save_data(power_spec, save_prefix+'Power-Spec-vs-Kappa',
                **data_array.attrs, **peak_info, stack_coords=True)

def process_power_spec_vs_time(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'cnoidal'},
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename)

        # Calculate power spectrum
        power_spec = power_spectrum(spatial_fourier_transform(
            data_array))

        # Convert from kappa*h to kappa/k coordinate
        power_spec = convert_k_E_to_k(power_spec)

        # Restrict to first num_peaks modes
        power_spec = power_spec[{'kappa/k':
            find_initial_peaks(power_spec, num_peaks=3)}]

        # Normalize by initial time
        power_spec = power_spec/power_spec[{'t*eps*sqrt(g*h)*k_E':0}]

        # Transpose to put the dependent data as the row index and for
        # convenience when viewing the CSV file (since there are
        # currently more columns than rows)
        power_spec = power_spec.transpose()

        # Save power spectrum
        data_csv.save_data(power_spec, save_prefix+'Power-Spec-vs-Time',
                **data_array.attrs, stack_coords=True)

def process_wavenumber_frequency(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'cnoidal'},
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename)

        # Calculate spatial fourier transform
        fourier = spatial_fourier_transform(data_array, repeat_times=20)

        # Convert from kappa*h to kappa/k coordinate
        fourier = convert_k_E_to_k(fourier)

        # Calculate temporal fourier transform
        wavenum_freq = temporal_fourier_transform(fourier)

        # Convert to power spectrum
        wavenum_freq = power_spectrum(wavenum_freq)

        # Save wavenumber-frequency data
        data_csv.save_data(wavenum_freq, save_prefix+'Wavenum-Freq',
                **data_array.attrs, stack_coords=True)

def process_depth_varying(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'DepthVarying'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    statistics_list = []
    k_Eh_val_list = []
    for filename in filenames:
        # Create shape statistics
        statistics = generate_statistics(filename)

        # Append statistics to list
        k_Eh_val_list.append(np.sqrt(float(statistics.attrs['mu'])))
        statistics_list.append(statistics)

    # Combine statistics Datasets
    statistics_datasets = xr.concat(statistics_list,
            dim=xr.DataArray(k_Eh_val_list, name='k_E*h', dims='k_E*h'))

    # Remove mu and wave_length parameter attributes
    statistics_datasets.attrs.pop('mu',None)
    statistics_datasets.attrs.pop('wave_length',None)

    # Choose final time
    statistics_datasets.attrs['t*eps*sqrt(g*h)*k_E'] = \
            statistics_datasets['t*eps*sqrt(g*h)*k_E'][-1].item()
    statistics_datasets = statistics_datasets[{'t*eps*sqrt(g*h)*k_E':-1}
            ].reset_coords(names='t*eps*sqrt(g*h)*k_E', drop=True)

    # Save statistics
    data_csv.save_data(statistics_datasets,
            save_prefix+'Shape-vs-Depth',
            **statistics_datasets.attrs)

def process_hofmiller(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Down sample time
        data_array = time_downsample(data_array,4)

        with xr.set_options(keep_attrs=True):
            # Speed in vertical offset per unit time
            speed = 0.1
            # Offset amplitudes by time
            data_array = data_array +\
                    data_array['t*eps*sqrt(g*h)*k_E']*speed

        # Save snapshots
        data_csv.save_data(data_array, save_prefix+'Hofmiller',
                **data_array.attrs, stack_coords=True)

def trim_snapshots(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Trim times
        data_array = time_fractions(data_array)

        # Save snapshots
        data_csv.save_data(data_array, save_prefix+'Snapshots',
                **data_array.attrs, stack_coords=True)

def move_verf(load_prefix, save_prefix, *args, **kwargs):
    # Simply move the verification data from the load directory to the
    # save directory unchanged

    # Find filenames
    old_filenames = glob.glob(load_prefix+'*'+'Verf'+'*')

    # Rename files
    new_filenames = [filename.replace(load_prefix, save_prefix) for
        filename in old_filenames]

    for old_filename,new_filename in zip(old_filenames, new_filenames):
        # Copy files
        shutil.copy(old_filename, new_filename)

def main():
    load_prefix = '../Data/Raw/'
    save_prefix = '../Data/Processed/'

    callable_functions = {
            'shape_statistics' : process_shape_statistics,
            'trim_snapshots' : trim_snapshots,
            'move_verf' : move_verf,
            'power_spec_vs_kappa' : process_power_spec_vs_kappa,
            'power_spec_vs_time' : process_power_spec_vs_time,
            'wavenum_freq' : process_wavenumber_frequency,
            'depth_varying' : process_depth_varying,
            'hofmiller' : process_hofmiller,
            }

    if len(sys.argv) == 1:
        # No option provided; run all plots
        for function in callable_functions.values():
            function(load_prefix, save_prefix)

    else:
        if not sys.argv[1] in callable_functions:
            raise ValueError('Command line option must be the name of '+
                'a callable function') from None
        if len(sys.argv) > 2:
            callable_functions[sys.argv[1]](load_prefix, save_prefix,
                    *sys.argv[2:])
        else:
            callable_functions[sys.argv[1]](load_prefix, save_prefix)

if __name__ == '__main__':
    main()
