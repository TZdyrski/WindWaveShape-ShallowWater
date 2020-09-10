#!/usr/bin/env python3
# postproccesor.py

# Note: this package requires xrscipy
# This package is not packaged by pypi and must be installed from
# source:
#   git clone https://github.com/fujiisoup/xr-scipy
#   cd xr-scipy
#   python setup.py install
import sys
import shutil
import glob
import scipy as sp
import scipy.integrate
import scipy.signal
import numpy as np
import scipy.special as spec
import xrscipy.signal as dsp
import xarray as xr
import data_csv
from useful_functions import round_sig_figs, derivative, get_var_stats

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

def asymmetry(profile, asym_half=None):
    """Get the asymmetry of the solution calculated with
    solve_system.

    Returns
    -------
    asymmetry : ndarray(t.size,)
        Returns the asymmetry calculated at each time step.
    """

    xLen,xNum,dx = get_var_stats(profile)

    # Create coordinate masks
    if asym_half == 'left':
        x_mask = (profile.coords['x/h']<0).values
    elif asym_half == 'right':
        x_mask = (profile.coords['x/h']>=0).values
    else:
        x_mask = np.ones(profile.coords['x/h'].size)

    # Cast to type double (float64) since fft (used in Hilbert for
    # asymmetry) is unable to hand long double (float128)
    profile = np.array(profile, dtype=np.float64)

    # Calculate the hilbert transform
    # Confusingly, sp.signal.hilbert gives the analytic signal, x -> x + i*H(x)
    # so take the imaginary part
    profile_hilbert = np.imag(sp.signal.hilbert(profile,axis=0))

    # Apply the coordinate mask (note: do this after Hilbert since
    # multiplying by a mask does not commute with the Hilbert transform)
    profile_hilbert = (profile_hilbert.T*x_mask).T

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

def slope(profile):
    _,_,dx = get_var_stats(profile)

    mu = profile.attrs['mu']

    # Calculate slope
    # Convert from
    # \partial eta'/\partial x' = \partial eta/(\partial x*h*k_E)
    # to
    # sqrt(mu) \partial eta'/\partial x' = \partial eta/(\partial x)
    # (Primes denote the nondim variables used throughout this solver)
    slope = derivative(profile, dx=dx,
            deriv_type='periodic_fd')*np.sqrt(mu)

    slope = xr.DataArray(slope, dims=profile.dims,
            coords=profile.coords, attrs=profile.attrs)

    return slope

def peak_location(profile):

    peak_locations = find_peak_coords(profile, dim_to_search='x/h',
            only_nonneg_coords=False, num_peaks=1, periodic=True)

    return peak_locations

def peak_speed(profile):

    # Find location of peaks
    peak_locations = peak_location(profile)

    # Calculate speed
    peak_speed = np.gradient(peak_locations)/np.gradient(profile['t*eps*sqrt(g*h)*k_E'])

    return peak_speed

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

    tLen, tNum, dt = get_var_stats(signal, var='t*eps*sqrt(g*h)*k_E',
            periodic=False)

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

    # Save values we will need for a calculation
    A = signal.attrs.get('A',1)
    B = signal.attrs.get('B',3/2)
    eps = signal.attrs.get('eps', 0.1)
    mu = signal.attrs.get('mu', 0.6)
    C = signal.attrs.get('C', 1/6*mu/eps)
    P = signal.attrs.get('P', 0.1)
    G = signal.attrs.get('G', -1/2*P*np.sqrt(mu/eps))
    H0 = signal.attrs.get('H0', 2)
    m = signal.attrs.get('m', H0*B/3/C)
    E = spec.ellipe(m)
    K = spec.ellipk(m)

    if G != 0:
        # Multiply each mode by (1+t/td) with
        # td = 45/4*A*C/G/B*m/H0*(m^2-m+2*E/K-2*E^2/K^2)/(-m^2+3*m-2+2*(m^2-m+1)*E/K)
        td = 45/4*A*C/G/B*m/H0*(m**2-m+2*E/K-2*E**2/K**2)/\
                (-m**2+3*m-2+2*(m**2-m+1)*E/K)
    else:
        # Explicitly define td to prevent a divide by zero error
        td = np.inf

    signal_modified = (1+signal['t*eps*sqrt(g*h)*k_E'].values/td)*signal

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
    power_spec_peak_indices = find_initial_peak_indices(power_spec, num_peaks=2)

    # Get indices for primary (m=1) and first harmonic (m=2)
    primaryIndex = power_spec_peak_indices[0]
    firstHarmonicIndex = power_spec_peak_indices[1]

    # Calculate bispectra
    # Note: this is the opposite sign convention to our
    # DeepWaterBiphaseManuscript; however, it matches the standard sign
    # convention, as well as that of Leykin et al 1995 and Feddersen and
    # Veron 2005
    bispectra = fourier[primaryIndex,:]**2*\
            np.conjugate(fourier[firstHarmonicIndex])

    # Calculate biphase
    biphase = np.angle(bispectra)

    return biphase

def bicoherence(profile):
    # Note: this function is not useful for deterministic outputs (like
    # ours) since the bicoherence is identically unity

    # Take spatial FFT \hat'{eps*eta'} = \hat{eps*eta'}*k_E = \hat{eta}*k_E/h
    # (Primes denote the nondim variables used throughout this solver)
    # (scale FFT by 1/N so the FFT of a sinusoid has unit amplitude;
    # this also gives it the same units as the continuous Fourier
    # Transform)
    fourier = spatial_fourier_transform(profile)

    fourier = convert_k_E_to_k(fourier)

    wrap_upper = fourier['kappa/k'].max()
    wrap_lower = fourier['kappa/k'].min()
    wrap_to_kappa = lambda x : (x - wrap_lower) % (wrap_upper -
            wrap_lower)+wrap_lower

    bispectra = xr.DataArray(None, dims=('kappa1/k','kappa2/k'),
        coords={'kappa1/k':fourier['kappa/k'].values,
        'kappa2/k':fourier['kappa/k'].values})
    bicoherence = bispectra

    # Calculate bispectra
    bispectra_eq =  lambda kappa1, kappa2 : fourier.loc[{'kappa/k' : kappa1}]*\
            fourier.loc[{'kappa/k' : kappa2}]*\
            np.conjugate(fourier.sel({'kappa/k' :
                wrap_to_kappa(kappa1+kappa2)}, method='nearest'))
    bispectra = bispectra_eq(bispectra['kappa1/k'],
            bispectra['kappa2/k'])

    # Calculate bicoherence
    bicoherence_eq = lambda kappa1,kappa2 : \
        np.abs(bispectra.sel({'kappa1/k':kappa1, 'kappa2/k':kappa2}))/\
                np.abs(fourier.sel({'kappa/k':kappa1}, method='nearest')*\
                fourier.sel({'kappa/k':kappa2}, method='nearest'))/\
                np.abs(fourier.sel({'kappa/k':wrap_to_kappa(kappa1+kappa2)},
                    method='nearest'))
    bicoherence = bicoherence_eq(bicoherence['kappa1/k'],\
            bicoherence['kappa2/k'])

    bicoherence.attrs = profile.attrs
    bicoherence.name = 'bicoherence'

    # Remove extraneous dimension
    bicoherence = bicoherence.reset_coords(names='kappa/k', drop=True)

    return bicoherence

def find_peak_coords(signal, dim_to_search='t*eps*sqrt(g*h)*k_E', num_peaks=5,
        only_nonneg_coords=True, center_domain=False, periodic=False):

    if only_nonneg_coords:
        # Only keep peaks where the corresponding dim_to_search
        # coordinate is nonnegative
        signal = signal.where(signal[dim_to_search] >= 0, drop=True)

    if center_domain:
        # Only look for peaks in the middle half of the (assumed)
        # half-open domain [a,b).
        # Useful when the domain contains two full waves, with peaks at
        # the left boundary, middle, and right boundary)
        domain = signal[dim_to_search].values

        _,_,dvar = get_var_stats(signal, var=dim_to_search)

        # We assume the domain is a half-open interval; temporarily add
        # the upper limit to simplify the calculations
        domain = np.append(domain, domain[-1]+dvar)

        domain_center = (domain.max()+domain.min())/2
        domain_range = (domain.max()-domain.min())
        new_domain_range = domain_range/2
        new_domain_upper_lim = domain_center+new_domain_range/2
        new_domain_lower_lim = domain_center-new_domain_range/2

        # Now remove the upper limit
        new_domain_upper_lim = new_domain_upper_lim - dvar

        signal = signal.where(np.logical_and(
            signal[dim_to_search] >= new_domain_lower_lim,
            signal[dim_to_search] < new_domain_upper_lim
            ), drop=True)


    def find_peak_indices(signal, num_peaks, periodic=False):
        # Need to do this to fix a bug in scipy
        signal = signal.copy()

        if periodic:
            # Wrap signal periodically
            signal = np.insert(signal, [0,len(signal)], [signal[-1], signal[0]])

        # Find peaks
        peak_indices = sp.signal.find_peaks(signal)[0]

        if periodic:
            # Decrease indices by one since we inserted an extra element at
            # the beginning while wrapping
            peak_indices = peak_indices - 1

        # Sort the peaks by the value of signal at that location
        peak_indices = peak_indices[np.argsort(
            signal[peak_indices]
            )[::-1]]

        # Only include num_peaks peaks
        peak_indices = peak_indices[:num_peaks]

        if peak_indices.size < num_peaks:
            raise(ValueError('Not enough peaks found; requested '+\
                    str(num_peaks)+' but only found '+\
                    str(peak_indices.size)))

        return peak_indices

    peak_indices = xr.apply_ufunc(
            find_peak_indices,
            signal,
            kwargs={'num_peaks':num_peaks,
                **({'periodic':periodic} if periodic is not None else
                    {})},
            input_core_dims=[[dim_to_search]],
            output_core_dims=[['peak_mag_high_to_low']],
            vectorize=True,
            )

    # Convert from indices to coordinates
    peak_coords = signal[dim_to_search][peak_indices]

    # Reduce number of dimensions
    peak_coords = peak_coords.squeeze(drop=True)

    # Remove extraneous dimension
    peak_coords = peak_coords.reset_coords(names=dim_to_search, drop=True)

    return peak_coords

def find_initial_peak_indices(signal, **kwargs):

    signal_initial = signal[{'t*eps*sqrt(g*h)*k_E':0}].reset_coords(
            't*eps*sqrt(g*h)*k_E', drop=True)

    # Determine the name of the coordinate (there should be only one
    # since we removed the time coordinate)
    coordinate = list(signal_initial.coords.keys())

    if len(coordinate) > 1:
        raise(ValueError("Too many dimensions when trying to find"+\
        "initial peak; there should only be the 't*eps*sqrt(g*h)*k_E"+\
        "dimension and one additional dimension"))
    else:
        coordinate = coordinate[0]

    # Find coordinates of peaks
    signal_peak_coords = find_peak_coords(signal_initial,
            dim_to_search=coordinate, **kwargs).values

    # Convert from coordinates to indices
    # Source: https://stackoverflow.com/questions/32191029/32191125#32191125
    sorter = np.argsort(signal_initial[coordinate].values)
    signal_peak_indices = sorter[np.searchsorted(
        signal_initial[coordinate].values,
        signal_peak_coords, sorter=sorter)]

    return signal_peak_indices

def truncate_after_peak(signal, num_peaks=3):
    signal_initial = np.abs(signal[{'t*eps*sqrt(g*h)*k_E':0}])

    # Only include first num_peaks+1 (we need one extra since the
    # right-sided base always gives the right window limit, so we'll use
    # the n+1th left base limit to define the nth right base limit)
    signal_peak_indices = find_initial_peak_indices(np.abs(signal), num_peaks=num_peaks+1)

    # Find peak bases in initial data (peaks shouldn't move over time either)
    signal_left_base_indices = sp.signal.peak_prominences(
            signal_initial, signal_peak_indices)[1]

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
    signal_peak_indices = find_initial_peak_indices(np.abs(signal), num_peaks=mode_num)

    # Store height (specifically, the largest height as a function of
    # time) and bases of second peak for inset image
    signal_peak_height = np.amax(signal[
        {'kappa/k':signal_peak_indices[mode_num-1]}]).item()

    # Find bases of mode_num-th peak in initial data (peaks shouldn't
    # move over time either) Calculate using 0.9999 prominence since we
    # want it cropped closely to the peak; the truncate_after_peak base
    # finding prescription would go from the left side of mode_num-th
    # peak to the left side of mode_num+1-th peak, including all the
    # intervening flat space.
    signal_peak_base_indices = sp.signal.peak_widths(
            signal_initial, [signal_peak_indices[mode_num-1]],
            rel_height=0.9999)[2:]

    # Convert from indices to kappa values by multiplying by the
    # kappa[1] (since all steps are evenly spaced)
    _,_,dkappa = get_var_stats(signal_initial, var='kappa/k',
            periodic=False)
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
    snapshot_fracs = np.array([0,3/8,3/4])

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

def time_downsample(signal, downsample_factor=1, keep_last=True):

    # Down sample the time component to save space since we don't need
    # the higher frequencies
    signal_downsampled = dsp.decimate(signal, downsample_factor, dim='t*eps*sqrt(g*h)*k_E')

    # Copy over attributes
    signal_downsampled.attrs = signal.attrs

    return signal_downsampled

def generate_statistics(filename=None, data_array=None, asymmetry_partitions=None):
    if filename is not None:
        # Extract data
        data_array = data_csv.load_data(filename)
    elif data_array is None:
        raise(ValueError('Must pass either filename or data_array to generate_statistics'))

    # Calculate max height
    maximums = maximum(data_array)

    # Calculate skewness
    skewnesses = skewness(data_array)

    # Calculate asymmetry
    asymmetries = asymmetry(data_array)
    if asymmetry_partitions:
        asymmetries_left = asymmetry(data_array, asym_half='left')
        asymmetries_right = asymmetry(data_array, asym_half='right')

    save_biphase = data_array.attrs.get('wave_type',None) == 'cnoidal'
    if save_biphase:
        # Calculate biphase
        biphases = biphase(data_array)

    # Calculate energy
    energies = energy(data_array)
    # Normalize by initial energy
    energies_normalized = energies/energies[{'t*eps*sqrt(g*h)*k_E':0}]

    # Calculate peak location
    peak_locations = peak_location(data_array)

    # Calculate speed of the peak
    peak_speeds = peak_speed(data_array)

    # Combine shape statistics
    statistics = xr.Dataset(data_vars = {
        'max(eta)/h' : ('t*eps*sqrt(g*h)*k_E' , maximums),
        'skewness' : ('t*eps*sqrt(g*h)*k_E' , skewnesses),
        'asymmetry' : ('t*eps*sqrt(g*h)*k_E' , asymmetries),
        **({'asymmetry_left' : ('t*eps*sqrt(g*h)*k_E' ,asymmetries_left),
            'asymmetry_right' : ('t*eps*sqrt(g*h)*k_E' ,asymmetries_right),
            } if asymmetry_partitions else {}),
        **({'biphase' : ('t*eps*sqrt(g*h)*k_E' , biphases)} if
            save_biphase else {}),
        'E/E_0' : ('t*eps*sqrt(g*h)*k_E' ,
            energies_normalized),
        'x_peak/h' : ('t*eps*sqrt(g*h)*k_E' , peak_locations),
        'c_peak/sqrt(g*h)' : ('t*eps*sqrt(g*h)*k_E' , peak_speeds),
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
            find_initial_peak_indices(power_spec, num_peaks=3)}]

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

def process_bicoherence(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'cnoidal'},
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename)

        # Calculate the bicoherence
        bicoh = bicoherence(data_array)

        # Save statistics
        data_csv.save_data(bicoh, save_prefix+'Bicoherence',
                **bicoh.attrs)

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

def process_press_varying(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'PressVarying'

    for wave_type in ['solitary', 'cnoidal']:
        # Find filenames
        filenames = data_csv.find_filenames(load_prefix, filename_base,
                allow_multiple_files=True,
                parameters={'wave_type' : wave_type})

        statistics_list = []
        P_val_list = []
        for filename in filenames:
            # Create shape statistics
            statistics = generate_statistics(filename)

            # Append statistics to list
            P_val_list.append(float(statistics.attrs['P']))
            statistics_list.append(statistics)

        # Combine statistics Datasets
        statistics_datasets = xr.concat(statistics_list,
                dim=xr.DataArray(P_val_list, name='P', dims='P'))

        # Remove P parameter attribute
        statistics_datasets.attrs.pop('P',None)

        # Sort by P
        statistics_datasets = statistics_datasets.sortby('P')

        # Choose final time
        statistics_datasets.attrs['t*eps*sqrt(g*h)*k_E'] = \
                statistics_datasets['t*eps*sqrt(g*h)*k_E'][-1].item()
        statistics_datasets = statistics_datasets[{'t*eps*sqrt(g*h)*k_E':-1}
                ].reset_coords(names='t*eps*sqrt(g*h)*k_E', drop=True)

        # Save statistics
        data_csv.save_data(statistics_datasets,
                save_prefix+'Shape-vs-Press',
                **statistics_datasets.attrs)

def process_xt_offset(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Down sample time
        num_lines = 20
        downsample_factor = int(round(
                data_array['t*eps*sqrt(g*h)*k_E'].size/num_lines))
        data_array = time_downsample(data_array,downsample_factor,
                keep_last=False)

        with xr.set_options(keep_attrs=True):
            # Speed in vertical offset per unit time
            speed = 0.1
            # Offset amplitudes by time
            data_array = data_array +\
                    data_array['t*eps*sqrt(g*h)*k_E']*speed

        # Save snapshots
        data_csv.save_data(data_array, save_prefix+'XT-Offset',
                **data_array.attrs, stack_coords=True)

def process_biviscosity_variation(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Biviscosity'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Create shape statistics
        statistics = generate_statistics(filename)

        # If nu_bi is float-0, replace it with int-0
        nu_bi = statistics.attrs['nu_bi']
        nu_bi = str(0 if nu_bi==0 else round_sig_figs(nu_bi,3))

        # Save statistics
        data_csv.save_data(statistics,
                save_prefix+'Biviscosity_nu_bi'+nu_bi,
                **statistics.attrs)

def process_decaying_no_nu_bi(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Decaying-no-NuBi'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Create shape statistics
        statistics = generate_statistics(filename)

        # Save statistics
        data_csv.save_data(statistics,
                save_prefix+'Decaying-no-NuBi', **statistics.attrs)

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

def process_spacetime_mesh(load_prefix, save_prefix, *args, **kwargs):
    # Simply move the snapshot data from the load directory to the save
    # directory unchanged
    filename_base = 'Snapshots'

    # Find filenames
    old_filenames = glob.glob(load_prefix+'*'+filename_base+'*')

    # Rename files
    new_filenames = [filename.\
            replace(load_prefix, save_prefix).\
            replace(filename_base, 'Full-'+filename_base)
            for filename in old_filenames]

    for old_filename,new_filename in zip(old_filenames, new_filenames):
        # Copy files
        shutil.copy(old_filename, new_filename)

def trim_trig_verf(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'TrigVerf'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Trim times
        data_array = time_fractions(data_array)

        # If nu_bi is float-0, replace it with int-0
        nu_bi = data_array.attrs['nu_bi']
        nu_bi = str(0 if nu_bi==0 else round_sig_figs(nu_bi,3))

        # Save snapshots
        data_csv.save_data(data_array,
                save_prefix+'TrigVerf'+data_array.attrs['solver']+\
                '_nu_bi'+str(nu_bi),
                **data_array.attrs, stack_coords=True)

def trim_long_verf(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongVerf'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Trim times
        data_array = time_fractions(data_array)

        # If nu_bi is float-0, replace it with int-0
        nu_bi = data_array.attrs['nu_bi']
        nu_bi = str(0 if nu_bi==0 else round_sig_figs(nu_bi,3))

        # Save snapshots
        data_csv.save_data(data_array,
                save_prefix+'LongVerf'+'_nu_bi'+str(nu_bi),
                **data_array.attrs, stack_coords=True)

def process_trig_statistics(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'TrigVerf'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Create shape statistics
        statistics = generate_statistics(filename)

        # If nu_bi is float-0, replace it with int-0
        nu_bi = statistics.attrs['nu_bi']
        nu_bi = str(0 if nu_bi==0 else round_sig_figs(nu_bi,3))

        # Save statistics
        data_csv.save_data(statistics, save_prefix+'TrigStatistics'+\
                '-'+statistics.attrs['solver']+'_nu_bi'+nu_bi,
                **statistics.attrs)

def process_long_statistics(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongVerf'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Create shape statistics
        statistics = generate_statistics(filename)

        # If nu_bi is float-0, replace it with int-0
        nu_bi = statistics.attrs['nu_bi']
        nu_bi = str(0 if nu_bi==0 else round_sig_figs(nu_bi,3))

        # Save statistics
        data_csv.save_data(statistics, save_prefix+'LongStatistics'+\
                '_nu_bi'+nu_bi,
                **statistics.attrs)

def process_snapshot_slopes(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Calculate slopes
        data_array = slope(data_array)

        # Trim times
        data_array = time_fractions(data_array)

        # Save snapshots
        data_csv.save_data(data_array, save_prefix+'Slopes',
                **data_array.attrs, stack_coords=True)

def process_slope_statistics(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Calculate slopes
        slopes = slope(data_array)

        # Create shape statistics
        statistics = generate_statistics(data_array=slopes)

        # Save statistics
        data_csv.save_data(statistics, save_prefix+'Slope-Statistics',
                **statistics.attrs)

def print_slope_ratios(load_prefix, save_prefix, *args, **kwargs):
    """
    Print ratio of (abs) highest slope peak to (abs) second-highest
    slope peak.
    """

    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    for filename in filenames:

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        if data_array.attrs['forcing_type'] == 'Jeffreys' and \
                data_array.attrs['mu'] == 0.6:

            # Calculate slopes
            data_array = slope(data_array)

            ## Print ratio of highest slope peak to second-highest
            peak_coords = find_peak_coords(np.abs(data_array), dim_to_search='x/h', num_peaks=2,
                    only_nonneg_coords=False)

            peaks = data_array.loc[{'x/h':peak_coords}]

            peak_ratios = np.abs(peaks[{'peak_mag_high_to_low':0}] / \
                    peaks[{'peak_mag_high_to_low':1}])

            print('Jeffreys forcing, solitary wave, P='+\
                    str(data_array.attrs['P']))
            print('Steepness ratio: '+str(peak_ratios.max().values))

def print_asym_variance(load_prefix, save_prefix, integrate_time=False,
        plot_variance=True, *args, **kwargs):
    """
    Determine what percent of the variance between
    abs(AsymPpos - AsymPzero) and abs(AsymPneg - AsymPzero) comes
    from the portion of the profile x > 0 compared to x < 0

    Parameters
    ----------
    integrate_time : boolean
        If True, integrate variance over time. If False, only take final
        time. Default is False.
    """

    filename_base = 'Snapshots'

    # Find filenames
    filenames = data_csv.find_filenames(load_prefix, filename_base,
            allow_multiple_files=True)

    statistics_list = []
    attrs_list = []
    for filename in filenames:
        # Create shape statistics
        statistics = generate_statistics(filename,
                asymmetry_partitions=True)

        if statistics.attrs['wave_type'] != 'solitary' or \
                statistics.attrs['forcing_type'] != 'Jeffreys':
                    continue

        # Append statistics to list
        statistics = statistics.assign_coords(statistics.attrs)
        statistics = statistics.expand_dims(list(statistics.attrs.keys()))
        statistics_list.append(statistics)

    # Combine statistics Datasets
    statistics_datasets = xr.combine_by_coords(statistics_list,
            combine_attrs='drop')
    statistics_datasets = statistics_datasets.squeeze(drop=True)

    # Loop through unique Ps, disregarding sign, and not including 0
    for P_abs_val in \
        set(np.abs(statistics_datasets.coords['P'].values)).difference([0]):

        variances = {}
        full_variances = {}

        for asym in ['asymmetry', 'asymmetry_left', 'asymmetry_right']:

            normalized_error = \
                    (np.abs( \
                        statistics_datasets[asym].loc[{'P':P_abs_val}] \
                        - statistics_datasets[asym].loc[{'P':0.0}]) \
                    - np.abs( \
                        statistics_datasets[asym].loc[{'P':-P_abs_val}] \
                        - statistics_datasets[asym].loc[{'P':0.0}]))

            # Take absolute value
            normalized_error = np.abs(normalized_error)
            if integrate_time:
                # At t=0, the denominator is zero, so normalized_error in inf;
                # replace with zero
                normalized_error[np.abs(normalized_error) == np.inf] = 0
                normalized_error = np.nan_to_num(normalized_error)

                tLen, tNum, dt = get_var_stats(statistics_datasets, var='t*eps*sqrt(g*h)*k_E',
                    periodic=False)
                integrated_variance = sp.integrate.trapz(normalized_error, dx=dt,
                        axis=0)/(tLen)

                variances[asym] = integrated_variance
            else:
                variances[asym] = normalized_error[{'t*eps*sqrt(g*h)*k_E':-1}].values
            full_variances[asym] = normalized_error

        print('abs(P) = '+str(P_abs_val))
        print('Left proportion of variance: '+\
                str(variances['asymmetry_left']/variances['asymmetry']))

        if plot_variance:
            import matplotlib.pyplot as plt
            time = statistics_datasets.coords['t*eps*sqrt(g*h)*k_E']
            variance_ratio = (full_variances['asymmetry_left'] \
                    /full_variances['asymmetry'])

            # Only show after t=1, since the early time has 'asymmetry',
            # 'asymmetry_left', and 'asymmetry_right' all approximately
            # zero, and the numerical precision causes issues
            time = time.where(time>1)
            variance_ratio = variance_ratio.where(time>1)

            plt.plot(time, variance_ratio)
            plt.title('Left proportion of variance: abs(P) = '+str(P_abs_val))
            plt.xlabel('t*eps*sqrt(g*h)*k_E')

            plt.show()


def main():
    load_prefix = '../Data/Raw/'
    save_prefix = '../Data/Processed/'

    callable_functions = {
            'shape_statistics' : process_shape_statistics,
            'trim_snapshots' : trim_snapshots,
            'trim_trig_verf' : trim_trig_verf,
            'trim_long_verf' : trim_long_verf,
            'trig_statistics' : process_trig_statistics,
            'long_statistics' : process_long_statistics,
            'snapshot_slopes' : process_snapshot_slopes,
            'slope_statistics' : process_slope_statistics,
            'power_spec_vs_kappa' : process_power_spec_vs_kappa,
            'power_spec_vs_time' : process_power_spec_vs_time,
            'wavenum_freq' : process_wavenumber_frequency,
            'depth_varying' : process_depth_varying,
            'press_varying' : process_press_varying,
            'xt_offset' : process_xt_offset,
            'biviscosity' : process_biviscosity_variation,
            'spacetime_mesh' : process_spacetime_mesh,
            'decaying_no_nu_bi' : process_decaying_no_nu_bi,
            'print_slope_ratios' : print_slope_ratios,
            'print_asym_variance' : print_asym_variance,
            }

    if len(sys.argv) == 1:
        # No option provided; run all plots except print functions
        for function in callable_functions.values():
            if function == print_slope_ratios or \
                    function == print_asym_variance:
                        continue
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
