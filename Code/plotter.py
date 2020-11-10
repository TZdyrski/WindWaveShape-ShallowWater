#!/usr/bin/env python3
# plotter.py
import sys
import numpy as np
import scipy.integrate
import xarray as xr
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.optimize import curve_fit, minimize
from pi_formatter import pi_multiple_ticks, float_to_pi
import itertools
import texplot
import data_csv
from useful_functions import round_sig_figs, get_var_stats

def property_cycle(num_lines,color_class='sequential'):
    """Generates a property cycle.

    Parameters
    ----------
    num_lines : int
        Number of lines in the cycle.
    color_class : 'sequential', 'inverse_sequential', or 'cyclic'
        Type of color class to use for lines. The 'sequential' option
        uses the 'YlGnBu' color class. The 'inverse_sequential' option
        uses the 'YlGnBu' color class in reverse. The 'cyclic' option
        uses the 'twilight' color class. Default is 'sequential'.

    Returns
    -------
    property_cycle : Cycle
        Property cycle with num_lines elements.

    """

    if color_class == 'sequential':
        # Add 1 to numerator and denominator so we don't go all the way to 0
        # (it's too light to see)
        new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
                range(num_lines)]

        linestyles = [*((0,(3+i,i)) for i in range(num_lines))]
    elif color_class == 'inverse_sequential':
        # Add 1 to numerator and denominator so we don't go all the way to 0
        # (it's too light to see)
        new_colors = [plt.get_cmap('viridis')(1. * (i)/(num_lines)) for i in
                reversed(range(num_lines))]

        linestyles = [*((0,(3+i,i)) for i in reversed(range(num_lines)))]
    elif color_class == 'cyclic':
        # Make the colors go from blue to black to red
        MaxColorAbs = 0.4

        new_colors = [plt.get_cmap('twilight')((i-num_lines/2+1/2)*2*
            MaxColorAbs/(num_lines+1)+0.5) for i in range(num_lines)]

        # Make the first half dotted
        linestyles = [*((0,(1,1+i)) for i in reversed(range(round((num_lines-1)/2))))]
        # Make the second half dashed (with the middle one solid)
        linestyles.extend([*((0,(3+i,i)) for i in range(round((num_lines+1)/2)))])
    else:
        raise(ValueError(
            "color_class must be either 'sequential', "\
                    +"inverse_sequential', or 'cyclic' but "\
                    + color_class+" was given"))

    prop_cycle = cycler('color', new_colors) + cycler('linestyle',
            linestyles)

    return prop_cycle

def annotate_arrow(ax, windLeft=True, wave_type='solitary'):
    if wave_type == 'solitary':
        arrowLeft = np.array([0.05,0.35])
        arrowRight = np.array([0.25,0.35])
        if windLeft:
            arrowLeft = arrowLeft + [0,0.1]
            arrowRight = arrowRight + [0,0.1]
            spacing = np.array([0,-0.225])
        else:
            spacing = np.array([0.7,0])
    elif wave_type == 'cnoidal':
        arrowLeft = np.array([0.025,0.35])
        arrowRight = np.array([0.225,0.35])
        if windLeft:
            arrowLeft = arrowLeft + [0,0.1]
            arrowRight = arrowRight + [0,0.1]
            spacing = np.array([0,-0.225])
        else:
            spacing = np.array([0.75,0])
    elif wave_type == 'tail':
        arrowLeft = np.array([0.05,0.1])
        arrowRight = np.array([0.25,0.1])
        spacing = np.array([0.7,0])
    else:
        raise(ValueError("'wave_type' must be either 'solitary' "+\
                "or 'cnoidal' but "+wave_type+" was given"))

    textBottom = (arrowLeft+arrowRight)/2 + np.array([0,0.05])

    # Annotate wave direction
    ax.annotate(r'Phase'+'\n'+r'Speed', xy=textBottom, xycoords='axes fraction',
            ha='center',va='bottom',ma='center')
    ax.annotate('', xy=arrowLeft, xytext=arrowRight,
            xycoords="axes fraction", arrowprops={'arrowstyle': '<-',
                'shrinkA':1,'shrinkB':0})

    # Annotate wind direction
    ax.annotate(r'Wind', xy=textBottom+spacing,
            xycoords='axes fraction', ha='center',va='bottom',
            ma='center')
    ax.annotate('', xy=arrowLeft+spacing, xytext=arrowRight+spacing,
            xycoords="axes fraction", arrowprops={
                'arrowstyle': ('<-' if windLeft else '->'),
                'shrinkA':1,'shrinkB':0})

def annotate_title(ax, title):
    textUpperLeft = (0.025,0.95)
    ax.annotate(title, xy=textUpperLeft, xycoords='axes fraction',
            ha='left',va='top',ma='left')

def label_subplots(ax, decrease_col_padding=False):

    if ax.size == 1:
        return
    for indx in np.ndindex(ax.shape):
        # Convert numerical index to lower case letter (0->a, 1->b, etc)
        indx_num = np.ravel_multi_index(indx,ax.shape)
        indxToAlpha = chr(indx_num+ord('a'))
        subplotLabel = indxToAlpha + ')'

        if (not decrease_col_padding) or indx[-1] == 0:
            padding = 0.275
        else:
            padding = 0.15

        # Add subplot labels
        t = ax[indx].text(-padding, 1.0, subplotLabel,
                transform=ax[indx].transAxes, va='top',ha='left')
        # Make box behind text semitransparent
        t.set_bbox(dict(facecolor='white', edgecolor='white',
            boxstyle='round'))

def atleast_2d(input_data):

    # Convert to array so we can transpose
    input_array = np.array(input_data)

    # Transpose, expand, then transpose back so that the new axis is
    # added after the existing one
    array_2d =  np.atleast_2d(input_array.T).T

    return array_2d

def fill_to_shape(scalar, shape):
    if np.isscalar(scalar):
        filled_array = np.empty(shape, dtype=object)
        filled_array.fill(scalar)
        return filled_array
    else:
        return scalar

def convert_x_norm(data_arrays):
    for iy,ix in np.ndindex(data_arrays.shape):
        # Convert from x'/sqrt(mu) = x/h to
        # x'/sqrt(mu)*sqrt(mu)/lambda' = x/lambda
        # (Primes denote the nondim variables used throughout this
        # solver)
        data_arrays[iy,ix] = data_arrays[iy,ix].assign_coords(
                {'x/lambda' : data_arrays[iy,ix]['x/h']*\
                        np.sqrt(float(data_arrays[iy,ix].attrs['mu']))/\
                        float(data_arrays[iy,ix].attrs['wave_length'])
                        })
        # Replace x/h with x/lambda as dimensional coordinate
        data_arrays[iy,ix] = data_arrays[iy,ix].swap_dims({'x/h' :
                'x/lambda'})
        # Remove x/h coordinate
        data_arrays[iy,ix] = data_arrays[iy,ix].drop('x/h')

    return data_arrays

def default_plotter(data_array, x_name, axis):
    axis.plot(data_array[x_name], data_array)

def plot_multiplot_template(data_arrays, x_coordinate, line_coord=None,
        suptitle=None, format_title=False, ax_title=None,
        ax_xlabel=None, ax_ylabel=None, color_class=None,
        decrease_col_padding=None,
        sort_lines=True, show_legend=False, legend_title=None,
        round_legend=True, plotter=default_plotter, label_plots=True,
        subplot_adjust_params={}, label_sig_figs=3, legend_sig_figs=2,
        trim_times=None, sharex=True, sharey='row',
        pi_parameters=[]):
    """
    Parameters
    ----------
    data_arrays : ndarray of xarrays
        An ndarray, dimension at most 2, with each element containing
        the xarray data_array for the corresponding subplot.
    """

    # Initialize figure
    fig, ax = texplot.newfig(1,nrows=data_arrays.shape[0],
            ncols=data_arrays.shape[1], sharex=sharex,sharey=sharey)

    # Make 2d ndarrays even if only scalars or 1d arrays
    ax = atleast_2d(ax)

    # If data_arrays has shape (1, n), then newfig will return a 1D
    # array of shape (n,). After expanding this to (n,1) with
    # atleast_2d, we need to transpose to ensure it has shape (n,1)
    if data_arrays.shape[0] == 1 and data_arrays.shape[1] > 1:
        ax = ax.T

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*(0.4+0.2*ax.shape[0])])

    # Save parameters into a separate array of dictionaries so we can
    # edit without affecting the dictionary
    parameters = np.empty(ax.shape, dtype=object)

    for iy, ix in np.ndindex(ax.shape):
        parameters[iy,ix] = data_arrays[iy,ix].attrs

        # Convert any the attrs dicts of data_arrays for all parameters in
        # pi_parameters
        for param in set(pi_parameters) & set(parameters[iy,ix].keys()):
            parameters[iy,ix][param] = \
            float_to_pi(float(parameters[iy,ix].get(param,'')))

        # Round parameters to correct number of significant figures
        parameters[iy,ix] = {k: round_sig_figs(v,label_sig_figs) if
                isinstance(v,float) else v for k,v in
                parameters[iy,ix].items()}

        # Label axes and titles
        if ax_xlabel is not None:
            axes_sharing = ax[iy,ix].get_shared_x_axes().get_siblings(ax[iy,ix])
            if iy == ax.shape[0]-1 or len(axes_sharing) == 1:
                ax[iy,ix].set_xlabel(fill_to_shape(ax_xlabel,ax.shape)[iy,ix]\
                        .format(**parameters[iy,ix]))
        if ax_ylabel is not None:
            axes_sharing = ax[iy,ix].get_shared_y_axes().get_siblings(ax[iy,ix])
            if ix == 0 or len(axes_sharing) == 1:
                ax[iy,ix].set_ylabel(fill_to_shape(ax_ylabel,ax.shape)[iy,ix]\
                        .format(**parameters[iy,ix]))
        if ax_title is not None:
            annotate_title(ax[iy,ix], fill_to_shape(ax_title,ax.shape)[iy,ix]\
                    .format(**parameters[iy,ix]))

        # Set property cycle
        prop_cycle = property_cycle(
                atleast_2d(data_arrays[iy,ix]).shape[1],
                **({'color_class':color_class} if color_class is not None
                    else {}),
                )
        ax[iy,ix].set_prop_cycle(prop_cycle)

        # Make 2d ndarrays even if only scalars or 1d arrays
        x_name = fill_to_shape(x_coordinate,ax.shape)[iy,ix]

        if atleast_2d(data_arrays[iy,ix]).shape[1] > 1 and sort_lines:
            if line_coord is None:
                # Sort lines
                line_coord = [val for val in data_arrays[iy,ix].dims if val
                        != x_name][0]
            data_array_sorted = \
                    data_arrays[iy,ix].sortby(line_coord)
        else:
            # Only a single line, no need to sort
            data_array_sorted = data_arrays[iy,ix]

        # Plot snapshots
        plotter(data_array_sorted, x_name, ax[iy,ix])

        # Trim xlim if x_coordinate == 't*eps*sqrt(g*h)*k_E' to
        # trim_times
        if x_coordinate == 't*eps*sqrt(g*h)*k_E' and \
                trim_times is not None:
            ax[iy,ix].set_xlim(right=trim_times)

    if suptitle is not None:
        if format_title:
            # Use values from first dataset since we assume they all use
            # the same values
            fig.suptitle(suptitle.format(**parameters[0,0]))
        else:
            fig.suptitle(suptitle)

    if label_plots:
        # Add subplot labels
        label_subplots(ax,
                **({'decrease_col_padding':decrease_col_padding} if
                    decrease_col_padding is not None else {}))

    if show_legend:
        # Add legend
        if sort_lines:
            # Use values from first data_array since we assume they all use the same
            # values
            line_labels=data_arrays[0,0].sortby(line_coord)[line_coord].values
        else:
            # Use values from first data_array since we assume they all use the same
            # values
            line_labels=data_arrays[0,0][line_coord].values
        if round_legend:
            line_labels=round_sig_figs(line_labels)

        leg = fig.legend(line_labels, **({'title':legend_title} if
            legend_title is not None else {}), loc='right')

        # Set alignment in case provided legend_title is multilined
        leg.get_title().set_multialignment('center')

    default_subplot_adjust_params = {'hspace':0, 'wspace':0.2}
    if suptitle is not None:
        default_subplot_adjust_params['top'] = 0.875

    if show_legend is not None:
        default_subplot_adjust_params['right'] = 0.825

    if default_subplot_adjust_params != {}:
        fig.set_tight_layout(False)
        fig.tight_layout()
        fig.subplots_adjust(**{**default_subplot_adjust_params,
            **subplot_adjust_params})

    # Make background transparent
    fig.patch.set_alpha(0)

    return fig

def plot_snapshots_template(data_arrays, norm_by_wavelength=True,
        wind_arrows=True, axes_single_col=True, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Distance $x/\lambda$' if norm_by_wavelength else\
            r'Distance $x/h$'

    ax_ylabel = 'Wave'+'\n'+r'profile $\eta / h$'

    title_string =  r'$\epsilon = {eps}$,'+\
            r' $\mu_E = {mu}$,'+\
            r' $P k_E/(\rho_w g \epsilon) = {P}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0,0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    # Convert psiP to fractions of pi
    pi_parameters = ['psiP']

    ax_title = np.empty(data_arrays.shape,dtype=object)
    for iy, ix in np.ndindex(ax_title.shape):
        if iy == 0:
            ax_title[0,ix] = title_string
        else:
            ax_title[iy,ix] = r'$P k_E/(\rho_w g \epsilon) = {P}$'

    x_coordinate = 'x/lambda' if norm_by_wavelength else 'x/h'

    legend_title = r'Time'+'\n'+r'$t \epsilon \sqrt{g h} k_E$'

    if norm_by_wavelength:
        data_arrays = convert_x_norm(data_arrays)

    # Reorder into a single column
    if axes_single_col:
        data_arrays = data_arrays.reshape((data_arrays.size,1), order='F')
        ax_title = ax_title.reshape((ax_title.size,1), order='F')

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays,
        'x_coordinate':x_coordinate,
        'pi_parameters':pi_parameters,
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        'show_legend':True,
        'legend_title':legend_title,
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    ax = atleast_2d(fig.axes).reshape(data_arrays.shape)

    # Zoom in on solitary wave
    for iy, ix in np.ndindex(ax.shape):
        if data_arrays[iy,ix].attrs.get('wave_type',None) == 'solitary':
            ax[iy,ix].set_xlim(-10,10)

    # Add arrow depicting wind and wave directions
    if wind_arrows:
        for iy, ix in np.ndindex(ax.shape):
            # Only put wind arrows if P is nonzero
            if data_arrays[iy,ix].attrs['P'] != 0:
                annotate_arrow(ax[iy,ix],
                        data_arrays[iy,ix].attrs['P']>0,
                        wave_type=data_arrays[iy,ix].attrs['wave_type'])

    return fig

def plot_snapshots_terms_template(data_arrays, **kwargs):

    # Use parameters from first data_array since we assume they're all
    # the same
    norm_by_wavelength = data_arrays[0,0].attrs['wave_type'] \
            == 'cnoidal'

    # Stack terms (generating a new coordinate 'variable' holding the
    # names of the data arrays)
    for indx in np.ndindex(data_arrays.shape):
        data_arrays[indx] = data_arrays[indx].to_array().T

    ax_ylabel = 'Terms'

    # Add more padding on right side for large legend
    subplot_adjust_params = {'right': 0.75}

    fig = plot_snapshots_template(**{
        'ax_ylabel':ax_ylabel,
        'data_arrays':data_arrays,
        'wind_arrows':False,
        'norm_by_wavelength':norm_by_wavelength,
        'subplot_adjust_params':subplot_adjust_params,
        'sort_lines':False,
        'round_legend':False,
        'line_coord':'variable',
        'legend_title':'',
        'axes_single_col':False,
        'decrease_col_padding':True,
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    ax = atleast_2d(fig.axes).reshape(data_arrays.shape)
    for iy,ix in np.ndindex(ax.shape):
        # Put horizontal line at y=0
        ax[iy,ix].axhline(0, color='0.75', zorder=-1)

    return fig

def plot_shape_statistics_template(data_arrays, ax_title=None, **kwargs):

    # Set axis labels and titles

    plot_biphase = 'biphase' in data_arrays[0].data_vars
    plot_peak_pos = 'x_peak/h' in data_arrays[0].data_vars
    plot_peak_speed = 'c_peak/sqrt(g*h)' in data_arrays[0].data_vars

    ax_ylabel_list = ['Energy'+'\n'+'$E/E_0$', 'Skewness'+'\n'+r'$\Sk/\Sk_0$',
            'Asymmetry'+'\n'+'$\As$']
    if plot_biphase:
        ax_ylabel_list.append('Biphase')
    if plot_peak_pos:
        # Insert after Energy
        ax_ylabel_list.insert(1,
                'Peak'+'\n'+'Position'+'\n'+r'$x_{{\text{{peak}}}}/h$')
    if plot_peak_speed:
        # Insert before Skewness
        ax_ylabel_list.insert(2,
                r'Peak Speed'+'\n'+r'$c_{{\text{{peak}}}}/\sqrt{{g h}}$')

    ax_ylabel = atleast_2d(np.array(ax_ylabel_list))

    if ax_title is not None:
        ax_title_full = fill_to_shape('', (len(ax_ylabel),data_arrays.size))
        ax_title_full[0,:] = ax_title

    data_arrays_rearranged = np.empty((ax_ylabel.size,data_arrays.size),dtype=object)

    for shape_group in range(data_arrays_rearranged.shape[1]):
        split_data_arrays = [
                data_arrays[shape_group]['E/E_0'],
                data_arrays[shape_group]['skewness']/
                data_arrays[shape_group]['skewness'][{kwargs.get('x_coordinate'):0}],
                data_arrays[shape_group]['asymmetry'],
                ]
        if plot_biphase:
            split_data_arrays.append(data_arrays[shape_group]['biphase'])
        if plot_peak_pos:
            # Insert after Energy
            split_data_arrays.insert(1,
                    data_arrays[shape_group]['x_peak/h'])
        if plot_peak_speed:
            # Insert before Skewness
            split_data_arrays.insert(2,
                    data_arrays[shape_group]['c_peak/sqrt(g*h)' ])

        data_arrays_rearranged[:,shape_group] = split_data_arrays
        for iy in range(data_arrays_rearranged.shape[0]):
            data_arrays_rearranged[iy,shape_group].attrs = \
                    data_arrays[shape_group].attrs

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays_rearranged,
        'ax_ylabel':ax_ylabel,
        **({'ax_title':ax_title_full} if ax_title is not None else {}),
        'color_class':'cyclic',
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    ax = atleast_2d(fig.axes).reshape((len(ax_ylabel_list),
        data_arrays.size))
    for ix in np.ndindex(ax.shape[1]):
        # Put horizontal line at y=1
        ax[ax_ylabel_list.index('Energy'+'\n'+'$E/E_0$'),ix].item().axhline(
                1, color='0.75', zorder=-1)

        # Put horizontal line at y=0
        ax[ax_ylabel_list.index('Asymmetry'+'\n'+r'$\As$'),ix].item().axhline(0,
                color='0.75', zorder=-1)

        if plot_peak_speed:
            # Put horizontal line at y=0
            ax[ax_ylabel_list.index(
                r'Peak Speed'+'\n'+r'$c_{{\text{{peak}}}}/\sqrt{{g h}}$'),
                    ix].item().axhline(0, color='0.75', zorder=-1)

        if plot_biphase:
            # Put horizontal line at y=0
            biphase_index = ax_ylabel_list.index('Biphase')
            ax[biphase_index,ix].item().axhline(0, color='0.75',
                    zorder=-1)

            # Determine smallest ylim (positive or negative)
            min_ylim = np.amin(np.abs(np.array(\
                    ax[biphase_index,ix].item().get_ylim())))
            # Don't let min_ylim be smaller than 1/2 max_ylim
            minSize = 1/2*np.amax(np.abs(np.array(\
                    ax[biphase_index,ix].item().get_ylim())))
            min_ylim = max([minSize,min_ylim])
            # Determine smallest power of 2, n, such that min_ylim >=
            # pi/2^n
            power_of_two = np.ceil(np.log(np.pi/min_ylim)/np.log(2))
            # Denominator of pi is 2^(power_of_two)
            pi_denom = 2**power_of_two
            # Set y-ticks as multiples of \pi
            pi_multiple_ticks(ax[biphase_index,ix].item(),'y',
                    1/pi_denom,1/(2*pi_denom))

    return fig

def plot_shape_statistics_vs_time_template(data_arrays, **kwargs):

    # Interpolate data
    for iy in range(data_arrays.shape[0]):
        data_arrays[iy] = data_arrays[iy].interpolate_na(
                't*eps*sqrt(g*h)*k_E', method='linear')

    # Set axis labels and titles

    ax_xlabel = r'Time $t \epsilon \sqrt{{g h}} k_E$'

    title_string =  r'$\epsilon = {eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    ax_title = np.empty(data_arrays.shape,dtype=object)
    for ix in np.ndindex(ax_title.shape):
        ax_title[ix] = title_string

    x_coordinate = 't*eps*sqrt(g*h)*k_E'

    legend_title = r'Pressure'+'\n'+r'Magnitude'+'\n'+\
                        r'$P k_E/(\rho_w g \epsilon)$'

    fig = plot_shape_statistics_template(data_arrays,
            **{
                'ax_xlabel':ax_xlabel,
                'ax_title':ax_title,
                'x_coordinate':x_coordinate,
                'show_legend':True,
                'legend_title':legend_title,
                'legend_sig_figs':3,
                **kwargs})

    return fig

def plot_shape_statistics_vs_depth_template(data_arrays, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Depth $k_E h$'

    title_string =  r'$\epsilon = {eps}$, '+\
            r'$t \epsilon \sqrt{{g h}} k_E = {t*eps*sqrt(g*h)*k_E}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    ax_title = np.empty(data_arrays.shape,dtype=object)
    for ix in np.ndindex(ax_title.shape):
        ax_title[ix] = title_string

    x_coordinate = 'k_E*h'

    fig = plot_shape_statistics_template(data_arrays,
            **{
                'ax_xlabel':ax_xlabel,
                'ax_title':ax_title,
                'x_coordinate':x_coordinate,
                'color_class':'cyclic',
                **kwargs})

    return fig

def plot_energy_template(data_arrays, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Time $t \epsilon \sqrt{{g h}} k_E$'

    ax_ylabel = r'Normalized Energy $Ek/(\rho_w g h^2)$'

    data_arrays_rearranged = np.empty((1,data_arrays.size),dtype=object)

    for shape_group in range(data_arrays_rearranged.shape[1]):
        split_data_arrays = [
                data_arrays[shape_group]['E/E_0'],
                ]

        data_arrays_rearranged[:,shape_group] = split_data_arrays
        for iy in range(data_arrays_rearranged.shape[0]):
            data_arrays_rearranged[iy,shape_group].attrs = \
                    data_arrays[shape_group].attrs

    # Use parameters from first data_array since we assume they're all
    # the same
    title_string = r'Total Energy: $\epsilon = {eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    # Convert psiP to fractions of pi
    pi_parameters = ['psiP']

    if data_arrays.size == 1:
        ax_title = title_string
        suptitle = None
    else:
        suptitle = title_string
        ax_title = None

    x_coordinate = 't*eps*sqrt(g*h)*k_E'

    # Plot P' = P*k/(rho_w*g)/eps
    # (Primes denote the nondim variables used throughout this solver)
    legend_title=r'Pressure'+'\n'+r'Magnitude'+'\n'+r'$P k_E/(\rho_w g \epsilon)$'

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays_rearranged,
        'x_coordinate':x_coordinate,
        'suptitle':suptitle,
        'format_title':True,
        'pi_parameters':pi_parameters,
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        'show_legend':True,
        'legend_title':legend_title,
        'legend_sig_figs':3,
        'color_class':'cyclic',
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    ax = atleast_2d(fig.axes)
    for ix in np.ndindex(ax.shape[1]):
        # Put horizontal line at y=1
        ax[0,ix].item().axhline(1, color='0.75', zorder=-1)

    return fig

def plot_power_spec_vs_kappa_template(data_arrays, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Harmonic $\kappa/k$'

    # Plot power spectrum \abs{\hat'{eps*eta'}}^2 = \abs{\hat{eta}}^2*k^2_E/h^2
    # (Primes denote the nondim variables used throughout this solver)
    ax_ylabel = r'Energy $\abs{{\hat{{\eta}}}}^2 k_E^2/h^2$'

    # Use parameters from first data_array since we assume they're all
    # the same
    suptitle = r'$\epsilon={eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0,0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    # Convert psiP to fractions of pi
    pi_parameters = ['psiP']

    # Convert P' = P*k/(rho_w*g)/eps to P'*eps = P*k/(rho_w*g)
    # (Primes denote the nondim variables used throughout this solver)
    title_string = r'$P k_E/(\rho_w g \epsilon) = {P}$'

    ax_title = np.empty(data_arrays.shape,dtype=object)
    for iy, ix in np.ndindex(ax_title.shape):
        if float(data_arrays[iy,ix].attrs['P']) >= 0:
            ax_title[iy,ix] = 'Co-Wind: '+title_string
        else:
            ax_title[iy,ix] = 'Counter-Wind: '+title_string

    x_coordinate = 'kappa/k'

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays,
        'x_coordinate':x_coordinate,
        'suptitle':suptitle,
        'format_title':True,
        'pi_parameters':pi_parameters,
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    ax = atleast_2d(fig.axes).reshape(data_arrays.shape)
    for iy,ix in np.ndindex(ax.shape):

        if float(data_arrays[iy,ix].attrs['P']) >= 0:
            zoom = 2.5
        else:
            zoom = 10

        # Put insets to zoom-in around second harmonic
        axins = zoomed_inset_axes(ax[iy,ix], zoom=zoom, loc=1)

        # Set property cycle
        prop_cycle = property_cycle(data_arrays[iy,ix].shape[1])
        axins.set_prop_cycle(prop_cycle)

        # Plot in inset
        axins.plot(data_arrays[iy,ix][x_coordinate],data_arrays[iy,ix])

        # Set inset limits
        axins.set_xlim(
                float(data_arrays[iy,ix].attrs['2-mode_left_base']),
                float(data_arrays[iy,ix].attrs['2-mode_right_base']),
                )
        axins.set_ylim(0,
                float(data_arrays[iy,ix].attrs['2-mode_height'])*1.05)

        # Add inset marking lines
        mark_inset(ax[iy,ix], axins, loc1=4, loc2=2, fc="none", ec="0.5")

        # Turn off tick labels
        axins.set_yticklabels([])
        axins.set_xticklabels([])

    # Use values from first dataset since we assume they all use the same
    # values
    leg = fig.legend(np.around(data_arrays[0,0]['t*eps*sqrt(g*h)*k_E'].values,1),
            title=r'Time'+'\n'+r'$t \epsilon \sqrt{g h} k_E$',
            loc='right')
    leg.get_title().set_multialignment('center')

    return fig

def plot_power_spec_vs_time_template(data_arrays, **kwargs):

    # Interpolate data
    for iy in range(data_arrays.shape[0]):
        data_arrays[iy] = data_arrays[iy].interpolate_na(
                't*eps*sqrt(g*h)*k_E', method='spline')

    # Set axis labels and titles

    ax_xlabel = r'Time $t \epsilon \sqrt{{g h}} k_E$'

    ax_ylabel = r'\begin{{tabular}}{{c}} Normalized \\'+\
            r'Energy $\abs{{\hat{{\eta}}}}^2 k_E^2/h^2$\end{{tabular}}'

    # Use parameters from first data_array since we assume they're all
    # the same
    suptitle = r'$\epsilon={eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0,0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    # Convert psiP to fractions of pi
    pi_parameters = ['psiP']

    title_string = r'$P k_E/(\rho_w g \epsilon) = {P}$'

    ax_title = np.empty(data_arrays.shape,dtype=object)
    for iy, ix in np.ndindex(ax_title.shape):
        if float(data_arrays[iy,ix].attrs['P']) >= 0:
            ax_title[iy,ix] = 'Co-Wind: '+title_string
        else:
            ax_title[iy,ix] = 'Counter-Wind: '+title_string

    x_coordinate = 't*eps*sqrt(g*h)*k_E'

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays,
        'x_coordinate':x_coordinate,
        'suptitle':suptitle,
        'format_title':True,
        'pi_parameters':pi_parameters,
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    # Add legend
    leg = fig.legend(['Primary', 'First\nHarmonic', 'Second\nHarmonic'],
            loc='right')
    leg.get_title().set_multialignment('center')


    return fig

def plot_wavenum_freq_template(data_arrays, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Harmonic $\kappa/k$'

    ax_ylabel = r'Frequency $\omega/(\epsilon \sqrt{{g h}} k_E)$'

    # Use parameters from first data_array since we assume they're all
    # the same
    suptitle = '$\epsilon = {eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0,0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    # Convert psiP to fractions of pi
    pi_parameters = ['psiP']

    title_string = r'$P k_E/(\rho_w g \epsilon) = {P}$'

    ax_title = np.empty(data_arrays.shape,dtype=object)
    for iy, ix in np.ndindex(ax_title.shape):
        if float(data_arrays[iy,ix].attrs['P']) >= 0:
            ax_title[iy,ix] = 'Co-Wind: '+title_string
        else:
            ax_title[iy,ix] = 'Counter-Wind: '+title_string

    x_coordinate = 'kappa/k'

    # Define custom plotter
    def contour_plotter(data_array, x_coord, axis):
        # Generate meshes for contour plot
        omega_mesh,kappa_mesh = np.meshgrid(
                data_array['omega/sqrt(g*h)/k/eps'],
                data_array['kappa/k'],
                )

        cs = axis.contourf(kappa_mesh, omega_mesh, data_array,
                locator=mpl.ticker.SymmetricalLogLocator(
                    linthresh=data_array.max()/1e6,base=10),
                norm=mpl.colors.SymLogNorm(
                    linthresh=data_array.max()/1e6,base=10),
                )

        # Adjust limits
        axis.set_xlim(-3.5,3.5)
        axis.set_ylim(-100,100)

        # Add colorbars
        axis.figure.colorbar(cs ,ax=axis,
                format=mpl.ticker.LogFormatterMathtext())
        return

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays,
        'x_coordinate':x_coordinate,
        'suptitle':suptitle,
        'format_title':True,
        'pi_parameters':pi_parameters,
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        'plotter':contour_plotter,
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    return fig

def plot_xt_offset_template(data_arrays, **kwargs):

    # Use parameters from first data_array since we assume they're all
    # the same
    norm_by_wavelength = data_arrays[0,0].attrs['wave_type'] \
            == 'cnoidal'

    def black_plotter(data_array, x_name, axis):
        axis.plot(data_array[x_name], data_array, linestyle='-', c='k')

    fig = plot_snapshots_template(**{
        'data_arrays':data_arrays,
        'show_legend':False,
        'wind_arrows':False,
        'plotter':black_plotter,
        'norm_by_wavelength':norm_by_wavelength,
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    # Put ticks on right side
    [ax.tick_params(right=True) for ax in fig.axes]

    return fig

def plot_spacetime_mesh_template(data_arrays, norm_by_wavelength=False,
        **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Time $t \epsilon \sqrt{{g h}} k_E$'

    ax_ylabel = r'Distance $x/\lambda$' if norm_by_wavelength else\
            r'Distance $x/h$'

    title_string =  r'$\epsilon = {eps}$,'+\
            r' $\mu_E = {mu}$,'+\
            r' $P k_E/(\rho_w g \epsilon) = {P}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0,0].attrs.get('forcing_type',None) ==
                    'GM' else '')+\
            (r', Forcing: {forcing_type}' if
                    data_arrays[0,0].attrs.get('forcing_type',None) is
                    not None else '')

    if data_arrays.size == 1:
        ax_title = title_string
        suptitle = None
    else:
        suptitle = title_string
        ax_title = None

    x_coordinate = 'x/lambda' if norm_by_wavelength else 'x/h'

    legend_title = r'Time'+'\n'+r'$t \epsilon \sqrt{g h} k_E$'

    # Convert psiP to fractions of pi
    pi_parameters = ['psiP']

    if norm_by_wavelength:
        data_arrays = convert_x_norm(data_arrays)

    # Define custom plotter
    def spacetime_mesh_plotter(data_array, x_coord, axis):

        # Generate meshes for contour plot
        t_mesh,x_mesh = np.meshgrid(
                data_array['t*eps*sqrt(g*h)*k_E'],
                data_array[x_coord],
                )

        cs = axis.pcolormesh(t_mesh, x_mesh, data_array,
                rasterized=True, shading='nearest')

        # Add colorbars
        axis.figure.colorbar(cs ,ax=axis)
        return

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays,
        'x_coordinate':x_coordinate,
        'suptitle':suptitle,
        'pi_parameters':pi_parameters,
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        'plotter':spacetime_mesh_plotter,
        'subplot_adjust_params':{'right':1},
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    return fig

def plot_metrics_mesh_template(data_arrays, norm_by_wavelength=False,
        **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'$x$-step $\Delta x$'

    ax_ylabel = r'$t$-step $\Delta t$'

    title_string =  r'$\epsilon = {eps}$,'+\
            r' $\mu_E = {mu}$,'+\
            r' $P k_E/(\rho_w g \epsilon) = {P}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0,0].attrs.get('forcing_type',None) ==
                    'GM' else '')+\
            (r', Forcing: {forcing_type}' if
                    data_arrays[0,0].attrs.get('forcing_type',None) is
                    not None else '')

    if data_arrays.size == 1:
        ax_title = title_string
        suptitle = None
    else:
        suptitle = title_string
        ax_title = None

    x_coordinate = 'xStep'

    # Convert psiP to fractions of pi
    pi_parameters = ['psiP']

    if norm_by_wavelength:
        data_arrays = convert_x_norm(data_arrays)

    # Define custom plotter
    def log_mesh_plotter(data_array, x_coord, axis):

        def face_center(X):
            """ Takes a 1D array X of size n and returns a 1D array X' of
            size n+1 with each point in X centered between two
            points in X' in log10 space. """
            # Source: matplotlib _axes.py
            X = np.log10(X)
            dX = np.diff(X)/2.
            X = np.hstack((X[0] - dX[0],
                X[:-1] + dX,
                X[-1] + dX[-1]))
            X = 10**X

            return X

        # Generate meshes for contour plot
        t_mesh,x_mesh = np.meshgrid(
                face_center(data_array['tStep']),
                face_center(data_array[x_coord]),
                )

        cs = axis.pcolormesh(x_mesh, t_mesh, data_array,
                rasterized=True,
                shading='flat',
        )

        axis.set_xscale('log')
        axis.set_yscale('log')

        # Add colorbars
        axis.figure.colorbar(cs ,ax=axis)

        return

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays,
        'x_coordinate':x_coordinate,
        'suptitle':suptitle,
        'pi_parameters':pi_parameters,
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        'plotter':log_mesh_plotter,
        'subplot_adjust_params':{'right':1},
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    return fig

def plot_forcing_types_template(data_arrays):
    # Initialize figure
    fig, ax = texplot.newfig(1,nrows=2,ncols=1,sharex=True,
            sharey=False)

    # Plot eta'/2 = eta/(2*a) = eta/H
    # (Primes denote the nondim variables used throughout this solver)
    ax[0].set_ylabel(r'Elevation $\eta/H$')
    ax[1].set_ylabel(r'\begin{tabular}{c}Pressure $p$\end{tabular}')

    # Plot x'*lambda' = x*k_E/lambda/k_E = x/lambda
    # (Primes denote the nondim variables used throughout this solver)
    ax[1].set_xlabel(r'Distance $x/\lambda$')

    ax[0].plot(data_arrays[0,0]['x/lambda'],data_arrays[0,0],color='#e7298a')
    ax[1].plot(data_arrays[1,0]['x/lambda'],data_arrays[1,0],label=r'Jeffreys $p_J$',
            color='#8da0cb',linestyle='dashed' )
    ax[1].plot(data_arrays[2,0]['x/lambda'],data_arrays[2,0],label=r'Generalized $p_G$',
            color='#fc8d62',linestyle='solid')

    # Add line depicting $\psi_P$
    WaveLength = data_arrays[0,0].attrs['WaveLength']
    psi_P = data_arrays[0,0].attrs['psi_P']
    ax[1].annotate('', xy=(WaveLength, 0), xytext=(WaveLength*(1-psi_P/(2*np.pi)), 0),
            arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5'))
    ax[1].annotate(r'$\psi_P$', xy=(WaveLength*(1-psi_P/(2*np.pi)/2.), 0), ha='center',
            va='bottom')

    # Add subplot labels
    label_subplots(ax)

    # Add legend
    fig.legend(loc='right', bbox_to_anchor=(0.525,0.5))

    # Make background transparent
    fig.patch.set_alpha(0)

    return fig

def plot_trig_verf_no_nu_bi(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'TrigVerf'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    # Extract data
    for indx_num, solver in enumerate(['Builtin', 'RK3']):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                required_words=[solver, 'nu_bi0'])

        data = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data

    # Make 2d ndarrays even if only scalars or 1d arrays
    data_arrays = np.atleast_2d(np.array(data_arrays))

    title_string = r'{solver} Solver after exactly 1 Period: '+\
                '$\epsilon = {eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0,0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    fig = plot_snapshots_template(data_arrays, suptitle=None,
            ax_ylabel=np.array([[''],['']]),
            ax_title=np.array([[title_string],[title_string]]),
            wind_arrows=False)

    texplot.savefig(fig,save_prefix+filename_base+'-no-NuBi')

def plot_trig_statistics_no_nu_bi(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'TrigStatistics'

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((2),dtype=object)

    solver_names = ['Builtin', 'RK3']

    # Extract data
    for indx_num, solver in enumerate(solver_names):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                required_words=[solver, 'nu_bi0'])

        data = data_csv.load_data(filename, stack_coords=False)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data

    title_string = r'Solver after exactly 1 Period: '+\
                '$\epsilon = {eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            suptitle = title_string,
            format_title = True,
            ax_title=solver_names,
            show_legend=False,
            )

    texplot.savefig(fig,save_prefix+filename_base+'-no-NuBi')

def plot_long_verf_solitary_no_nu_bi(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongVerf'

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'solitary'}, required_words=['nu_bi0'])

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Long-Run-no-NuBi')

def plot_long_verf_cnoidal_no_nu_bi(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongVerf'

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'cnoidal'}, required_words=['nu_bi0'])

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Long-Run-Cnoidal-no-NuBi')

def plot_long_statistics_solitary_no_nu_bi(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongStatistics'

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)

    # Extract data
    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'solitary'},
            required_words=['nu_bi0'])

    data_arrays[0] = data_csv.load_data(filename, stack_coords=False)

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            show_legend=False,
            )

    texplot.savefig(fig,save_prefix+filename_base+'-Solitary-no-NuBi')

def plot_long_statistics_cnoidal_no_nu_bi(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongStatistics'

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)

    # Extract data
    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'cnoidal'},
            required_words=['nu_bi0'])

    data_arrays[0] = data_csv.load_data(filename, stack_coords=False)

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            show_legend=False,
            )

    texplot.savefig(fig,save_prefix+filename_base+'-Cnoidal-no-NuBi')

def plot_trig_verf(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'TrigVerf'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    # Extract data
    for indx_num, solver in enumerate(['Builtin', 'RK3']):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                required_words=[solver], forbidden_words=['nu_bi0'])

        data = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data

    # Make 2d ndarrays even if only scalars or 1d arrays
    data_arrays = np.atleast_2d(np.array(data_arrays))

    title_string = r'{solver} Solver after exactly 1 Period: '+\
                '$\epsilon = {eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0,0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    fig = plot_snapshots_template(data_arrays, suptitle=None,
            ax_ylabel=np.array([[''],['']]),
            ax_title=np.array([[title_string],[title_string]]),
            wind_arrows=False)

    texplot.savefig(fig,save_prefix+filename_base)

def plot_trig_statistics(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'TrigStatistics'

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((2),dtype=object)

    solver_names = ['Builtin', 'RK3']

    # Extract data
    for indx_num, solver in enumerate(solver_names):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                forbidden_words=[solver, 'nu_bi0'])

        data = data_csv.load_data(filename, stack_coords=False)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data

    title_string = r'Solver after exactly 1 Period: '+\
                '$\epsilon = {eps}$, $\mu_E = {mu}$'+\
            (r', $\psi_P = {psiP}$' if
                    data_arrays[0].attrs.get('forcing_type',None) ==
                    'GM' else '')

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            suptitle = title_string,
            format_title = True,
            ax_title=solver_names,
            show_legend=False,
            )

    texplot.savefig(fig,save_prefix+filename_base)

def plot_long_statistics_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongStatistics'

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)

    # Extract data
    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'solitary'},
            forbidden_words=['nu_bi0'])

    data_arrays[0] = data_csv.load_data(filename, stack_coords=False)

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            show_legend=False,
            )

    texplot.savefig(fig,save_prefix+filename_base+'-Solitary')

def plot_long_statistics_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongStatistics'

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)

    # Extract data
    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'cnoidal'},
            forbidden_words=['nu_bi0'])

    data_arrays[0] = data_csv.load_data(filename, stack_coords=False)

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            show_legend=False,
            )

    texplot.savefig(fig,save_prefix+filename_base+'-Cnoidal')

def plot_long_verf_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongVerf'

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'solitary'}, forbidden_words=['nu_bi0'])

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Long-Run')

def plot_long_verf_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongVerf'

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'cnoidal'}, forbidden_words=['nu_bi0'])

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Long-Run-Cnoidal')

def plot_verf_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'solitary', 'P':0})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    # Plot points, not lines
    def point_plotter(data_array, x_name, axis):
        axis.plot(data_array[x_name], data_array,'.')

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False,
            plotter=point_plotter)

    texplot.savefig(fig,save_prefix+'Solitary-Verf')

def plot_pos_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Snapshots')

def plot_neg_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    kwargs['P'] = -kwargs['P']
    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Snapshots-Negative')

def plot_pos_neg_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs,
                'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative')

def plot_pos_neg_solitary_production(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs,
                'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    ax_title=np.array([[r'$P k_E/(\rho_w g \epsilon) = {P}$'],
        [r'$P k_E/(\rho_w g \epsilon) = {P}$']])

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False,
            ax_title=ax_title)

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative-Production')

def plot_pos_neg_solitary_terms(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Terms'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,2),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs,
                'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=False,
                coord_names=['x/h','t*eps*sqrt(g*h)*k_E'])
        data_array = data_array.squeeze(dim='t*eps*sqrt(g*h)*k_E')

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

        xLen,_,dx = get_var_stats(data_arrays[indx])

        print('RMS for P='+str(P_val)+' of term:')
        minRMS = np.inf
        minTerm = ''
        hyperviscRMS = 0
        for term in data_arrays[indx]:
            L2terms = scipy.integrate.trapz(
                    data_arrays[indx][term]**2, dx=dx,
                    axis=0)/(xLen)
            RMSterms = np.sqrt(L2terms)
            print('    '+term+': '+str(RMSterms))
            if term != 'Hyperviscosity' and RMSterms < minRMS:
                minRMS = RMSterms
                minTerm = term
            if term == 'Hyperviscosity':
                hyperviscRMS = RMSterms
        print('    '+'Ratio of '+minTerm+' to hyperviscosity: '+\
                str(minRMS/hyperviscRMS))

        print('Range for P='+str(P_val)+' of term:')
        for term in data_arrays[indx]:
            rangeTerms = data_arrays[indx][term].max() - \
                    data_arrays[indx][term].min()
            print('    '+term+': '+str(rangeTerms.values))

        indx = tuple(map(sum, zip(indx,(1,0))))
        data_array = data_array.copy()
        data_array['Current'] = None
        data_array['Advection'] = None
        data_array['Dispersion'] = None
        data_arrays[indx] = data_array

    ax_title=np.array([[r'$P k_E/(\rho_w g \epsilon) = {P}$',r'$P k_E/(\rho_w g \epsilon) = {P}$'],
        [r'$P k_E/(\rho_w g \epsilon) = {P}$',r'$P k_E/(\rho_w g \epsilon) = {P}$']])

    fig = plot_snapshots_terms_template(data_arrays, norm_by_wavelength=False,
            ax_title=ax_title)

    texplot.savefig(fig,save_prefix+'Terms-Positive-Negative')

def print_solitary_unforced_difference(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    P = 0

    filename = data_csv.find_filenames(load_prefix, filename_base,
        parameters={'wave_type' : 'solitary', **kwargs,
            'P' : P})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Subtract last time from first time
    difference = data_array[{'t*eps*sqrt(g*h)*k_E':-1}] - \
            data_array[{'t*eps*sqrt(g*h)*k_E':0}]

    xLen,_,dx = get_var_stats(data_array)

    # Take the L2 norm
    L2diff = scipy.integrate.trapz(difference**2, dx=dx,
            axis=0)/(xLen)

    # Normalize by the original L2 norm
    L2orig = scipy.integrate.trapz(
            data_array[{'t*eps*sqrt(g*h)*k_E':0}]**2, dx=dx,
            axis=0)/(xLen)
    L2ratio = np.sqrt(L2diff/L2orig)

    print('Normalized, RMS between end and start for '\
            +'unforced solitary wave:'+str(L2ratio))

    maximum = data_array.max(dim='x/h')
    minimum = data_array.min(dim='x/h')
    height = maximum - minimum

    heightRatio = height[{'t*eps*sqrt(g*h)*k_E':-1}]/\
            height[{'t*eps*sqrt(g*h)*k_E':0}]
    print('One minus ratio of final height (max-min) to initial height:'+\
            str(1-heightRatio.values))

def fit_sech(profile):

    x = profile['x/h']
    t = profile['t*eps*sqrt(g*h)*k_E']
    eps = profile.attrs['eps']
    mu = profile.attrs['mu']

    # Fit with sech^2
    H = np.empty(t.size)
    x_center = np.empty(t.size)
    for elem,_ in enumerate(t):
        def assumed_sech(x,H,x_center):
            return eps*H/np.cosh(np.sqrt(H/8)*(x-x_center)*np.sqrt(mu))**2

        def cost(params, x, result):
            H,x_center = params
            model = assumed_sech(x,H,x_center)
            cost = np.sum(np.abs(result - model))
            return cost

        max_loc = profile[{'t*eps*sqrt(g*h)*k_E':elem}].idxmax()
        max_height = profile[{'t*eps*sqrt(g*h)*k_E':elem}].max()
        result = minimize(cost, x0=(max_height,max_loc),
                args=(x.values,
                    profile[{'t*eps*sqrt(g*h)*k_E':elem}].values),
                bounds=((0,None),(None,None)))

        H[elem] = result.x[0] # store height
        x_center[elem] = result.x[1] # store center location

    # Convert H and x_center to DataArrays
    H = xr.DataArray(H,
            dims=['t*eps*sqrt(g*h)*k_E'],
            coords={'t*eps*sqrt(g*h)*k_E': profile['t*eps*sqrt(g*h)*k_E']})
    x_center = xr.DataArray(x_center,
            dims=['t*eps*sqrt(g*h)*k_E'],
            coords={'t*eps*sqrt(g*h)*k_E': profile['t*eps*sqrt(g*h)*k_E']})

    return H, x_center

def symmetric_approximation(profile):
    ## Calculate approximate symmetric, sech^2 profile

    # Calculate maximum
    eps = profile.attrs['eps']
    mu =  profile.attrs['mu']

    # Generate sech-centered coordinate
    profile = sech_centered_coord(profile)

    # Get profile height
    H,_ = fit_sech(profile)

    # Create symmetric, sech^2 profile
    symmetric_approx = eps*H/np.cosh(np.sqrt(H/8)*\
            (profile.coords['(x-x_center)/h']*np.sqrt(mu)))**2

    return symmetric_approx

def sech_centered_coord(profile):
    # Get center of sech^2
    _,x_center = fit_sech(profile)

    # Boost into sech-centered frame
    profile = profile.assign_coords({'(x-x_center)/h':
        profile.coords['x/h']-x_center})

    return profile

def plot_pos_neg_solitary_tail(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        parameters = parameters={'wave_type' : 'solitary', **kwargs,
                'P' : P_val}
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters=parameters)

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Create symmetric approximation
        symmetric_approx = symmetric_approximation(data_array)
        data_array.attrs['wave_type'] = 'tail'

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array - symmetric_approx
        data_arrays[indx].attrs = data_array.attrs

    ax_title=np.array([[r'$P k_E/(\rho_w g \epsilon) = {P}$'],
        [r'$P k_E/(\rho_w g \epsilon) = {P}$']])
    ax_xlabel=r'Reference wave-centered distance $\tilde{{x}}/h$'
    ax_ylabel='Profile'+'\n'+r'change $\Delta \eta/h$'

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False,
            ax_title=ax_title,
            ax_xlabel=ax_xlabel,
            ax_ylabel=ax_ylabel,
            x_coordinate='(x-x_center)/h',
            line_coord='t*eps*sqrt(g*h)*k_E',
            sharey=True)

    # Zoom in on wave
    for ax in fig.axes:
        ax.set_xlim(right=15)

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative-Tail')

def plot_pos_neg_solitary_tail_thumbnail(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)

    P = float(kwargs.get('P'))

    parameters = parameters={'wave_type' : 'solitary', **kwargs,
            'P' : P}
    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters=parameters)

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Create symmetric approximation
    symmetric_approx = symmetric_approximation(data_array)
    data_array.attrs['wave_type'] = 'tail'

    data_arrays[0,0] = data_array - symmetric_approx
    data_arrays[0,0].attrs = data_array.attrs
    # Only show first and last
    data_arrays[0,0] = data_arrays[0,0].where(
            (data_arrays[0,0].coords['t*eps*sqrt(g*h)*k_E'] < 1)
            | (data_arrays[0,0].coords['t*eps*sqrt(g*h)*k_E'] > 2)
            ,drop=True)

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False,
            ax_title='',
            ax_xlabel='',
            ax_ylabel='',
            x_coordinate='(x-x_center)/h',
            line_coord='t*eps*sqrt(g*h)*k_E',
            show_legend=False,
            wind_arrows=False)

    fig_size_cm = np.array([2.4,2])
    fig_size_in = fig_size_cm/2.54
    fig.set_size_inches(fig_size_in)

    # Zoom in on wave
    for ax in fig.axes:
        ax.set_xlim(right=10)
        ax.axis('off')
        lines = ax.get_lines()
        for line in lines:
            line.set_linestyle('-')

    fig.set_tight_layout({'pad':0.1})

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative-Tail-Thumbnail')

def plot_pos_neg_solitary_and_sech(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        parameters={'wave_type' : 'solitary', **kwargs, 'P' : P_val}
        filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters=parameters)

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Create symmetric approximation
        symmetric_approx = symmetric_approximation(data_array)
        data_array.attrs['wave_type'] = 'tail'

        # Generate sech-centered coordinate
        data_array = sech_centered_coord(data_array)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = xr.concat([
            data_array, symmetric_approx], dim='t*eps*sqrt(g*h)*k_E')
        data_arrays[indx].attrs = data_array.attrs

    ax_title=np.array([[r'$P k_E/(\rho_w g \epsilon) = {P}$'],
        [r'$P k_E/(\rho_w g \epsilon) = {P}$']])
    ax_xlabel=r'Reference wave-centered distance $\tilde{{x}}/h$'

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False,
            line_coord='t*eps*sqrt(g*h)*k_E',
            x_coordinate='(x-x_center)/h',
            ax_xlabel=ax_xlabel,
            ax_title=ax_title)

    # Zoom in on crest
    for ax in fig.axes:
        ax.set_xlim([-3,3])

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative-and-Sech')

def plot_pos_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'cnoidal', **kwargs})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Snapshots-Cnoidal')

def plot_neg_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    kwargs['P'] = -kwargs['P']
    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'cnoidal', **kwargs})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Snapshots-Negative-Cnoidal')

def plot_pos_neg_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,2),dtype=object)

    mu = float(kwargs.get('mu'))
    P = float(kwargs.get('P'))

    for indx_num, (P_val, mu_val) in enumerate(itertools.product([P,-P],
        round_sig_figs([mu,7/8*mu]))):

        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val, 'mu' : mu_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_snapshots_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative-Cnoidal')

def plot_pos_neg_cnoidal_GM(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Snapshots'

    kwargs['forcing_type'] = 'GM'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_snapshots_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative-Cnoidal-GM')

def plot_pos_neg_slope_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Slopes'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs,
                'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = np.abs(data_array)
        data_arrays[indx].attrs = data_array.attrs

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False,
            ax_ylabel=r'Wave Slope $\abs{{\partial \eta/ \partial x}}$')

    texplot.savefig(fig,save_prefix+'Slopes-Positive-Negative')

def plot_pos_neg_slope_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Slopes'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,2),dtype=object)

    mu = float(kwargs.get('mu'))
    P = float(kwargs.get('P'))

    for indx_num, (P_val, mu_val) in enumerate(itertools.product([P,-P],
        round_sig_figs([mu,7/8*mu]))):

        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val, 'mu' : mu_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = np.abs(data_array)
        data_arrays[indx].attrs = data_array.attrs

    fig = plot_snapshots_template(data_arrays,
            ax_ylabel=r'Wave Slope $\abs{{\partial \eta/ \partial x}}$')

    texplot.savefig(fig,save_prefix+'Slopes-Positive-Negative-Cnoidal')

def plot_pos_neg_slope_cnoidal_GM(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Slopes'

    kwargs['forcing_type'] = 'GM'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = np.abs(data_array)
        data_arrays[indx].attrs = data_array.attrs

    fig = plot_snapshots_template(data_arrays,
            ax_ylabel=r'Wave Slope $\abs{{\partial \eta/ \partial x}}$')

    texplot.savefig(fig,save_prefix+'Slopes-Positive-Negative-Cnoidal-GM')

def plot_slope_statistics_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Slope-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs},
            allow_multiple_files=True)

    data_array_list = []
    P_val_list = []
    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=False)

        P_val = float(data_array.attrs['P'])

        P_val_list.append(P_val)
        data_array_list.append(data_array)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = xr.concat(data_array_list, dim=xr.DataArray(P_val_list,
        name='P', dims='P'))
    # Transpose to put P coordinate at end; this makes plotting easier
    data_arrays[0] = data_arrays[0].transpose()
    # Remove P parameter attribute
    data_arrays[0].attrs.pop('P', None)

    fig = plot_shape_statistics_vs_time_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Slope-Skew-Asymm')

def plot_slope_statistics_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Slope-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((2),dtype=object)

    mu = float(kwargs.get('mu'))

    for indx_num, mu_val in enumerate(round_sig_figs([mu,7/8*mu])):
        filenames = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'mu' : mu_val},
                allow_multiple_files=True)

        data_array_list = []
        P_val_list = []
        for filename in filenames:
            # Extract data
            data_array = data_csv.load_data(filename, stack_coords=False)

            P_val = float(data_array.attrs['P'])

            P_val_list.append(P_val)
            data_array_list.append(data_array)

        data_arrays[indx_num] = xr.concat(data_array_list,
                dim=xr.DataArray(P_val_list, name='P', dims='P'))
        # Transpose to put P coordinate at end; this makes plotting easier
        data_arrays[indx_num] = data_arrays[indx_num].transpose()
        # Remove P parameter attribute
        data_arrays[indx_num].attrs.pop('P', None)

    fig = plot_shape_statistics_vs_time_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Slope-Skew-Asymm-Cnoidal')

def plot_shape_statistics_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs},
            allow_multiple_files=True)

    data_array_list = []
    P_val_list = []
    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=False)

        P_val = float(data_array.attrs['P'])

        P_val_list.append(P_val)
        data_array_list.append(data_array)

        if P_val == 0:
            # Show ratio of final unforced energy to initial unforced
            # energy
            energy_change = data_array['E/E_0'][{'t*eps*sqrt(g*h)*k_E':-1}] \
                    / data_array['E/E_0'][{'t*eps*sqrt(g*h)*k_E':0}]
            print('One minus ratio of final unforced energy to '+\
                    'initial energy:'+str(1-energy_change.values))

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = xr.concat(data_array_list, dim=xr.DataArray(P_val_list,
        name='P', dims='P'))
    # Transpose to put P coordinate at end; this makes plotting easier
    data_arrays[0] = data_arrays[0].transpose()
    # Remove P parameter attribute
    data_arrays[0].attrs.pop('P', None)

    fig = plot_shape_statistics_vs_time_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Skew-Asymm')

def plot_shape_statistics_solitary_production(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs},
            allow_multiple_files=True)

    data_array_list = []
    P_val_list = []
    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=False)

        P_val = float(data_array.attrs['P'])

        P_val_list.append(P_val)

        # Remove peak position
        data_array = data_array.drop('x_peak/h')

        # Remove peak speed
        data_array = data_array.drop('c_peak/sqrt(g*h)')

        data_array_list.append(data_array)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = xr.concat(data_array_list, dim=xr.DataArray(P_val_list,
        name='P', dims='P'))
    # Transpose to put P coordinate at end; this makes plotting easier
    data_arrays[0] = data_arrays[0].transpose()
    # Remove P parameter attribute
    data_arrays[0].attrs.pop('P', None)

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            ax_title=None)

    texplot.savefig(fig,save_prefix+'Skew-Asymm-Production')

def plot_shape_statistics_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((2),dtype=object)

    mu = float(kwargs.get('mu'))

    for indx_num, mu_val in enumerate(round_sig_figs([mu,7/8*mu])):
        filenames = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'mu' : mu_val},
                allow_multiple_files=True)

        data_array_list = []
        P_val_list = []
        for filename in filenames:
            # Extract data
            data_array = data_csv.load_data(filename, stack_coords=False)

            P_val = float(data_array.attrs['P'])

            P_val_list.append(P_val)
            data_array_list.append(data_array)

        data_arrays[indx_num] = xr.concat(data_array_list,
                dim=xr.DataArray(P_val_list, name='P', dims='P'))
        # Transpose to put P coordinate at end; this makes plotting easier
        data_arrays[indx_num] = data_arrays[indx_num].transpose()
        # Remove P parameter attribute
        data_arrays[indx_num].attrs.pop('P', None)

    fig = plot_shape_statistics_vs_time_template(data_arrays)

    ax = fig.axes[0]
    t = data_arrays[0]['t*eps*sqrt(g*h)*k_E']
    initial_H = 2*data_arrays[0].attrs['eps']
    energy = initial_H*np.exp(np.outer(t,P_val_list)*0.3)
    ax.plot(t, energy, color='y', zorder=-1)

    texplot.savefig(fig,save_prefix+'Skew-Asymm-Cnoidal')

def plot_shape_statistics_cnoidal_GM(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-Statistics'

    kwargs['forcing_type'] = 'GM'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((2),dtype=object)

    mu = float(kwargs.get('mu'))

    for indx_num, mu_val in enumerate(round_sig_figs([mu,7/8*mu])):
        filenames = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'mu' : mu_val},
                allow_multiple_files=True)

        data_array_list = []
        P_val_list = []
        for filename in filenames:
            # Extract data
            data_array = data_csv.load_data(filename, stack_coords=False)

            P_val = float(data_array.attrs['P'])

            P_val_list.append(P_val)
            data_array_list.append(data_array)

        data_arrays[indx_num] = xr.concat(data_array_list,
                dim=xr.DataArray(P_val_list, name='P', dims='P'))
        # Transpose to put P coordinate at end; this makes plotting easier
        data_arrays[indx_num] = data_arrays[indx_num].transpose()
        # Remove P parameter attribute
        data_arrays[indx_num].attrs.pop('P', None)

    fig = plot_shape_statistics_vs_time_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Skew-Asymm-Cnoidal-GM')

def plot_shape_statistics_vs_depth(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-vs-Depth'

    # Remove 'mu' parameter
    kwargs.pop('mu', None)

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'cnoidal', **kwargs})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=False)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = data_array

    fig = plot_shape_statistics_vs_depth_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Skew-Asymm-Cnoidal-kh')

def plot_shape_statistics_vs_press_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-vs-Press'

    # Remove 'mu' parameter
    kwargs.pop('P', None)

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=False)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = data_array

    fig = plot_shape_statistics_vs_depth_template(data_arrays,
            x_coordinate = 'P',
            ax_xlabel = r'Pressure $P k/(\rho_w g \epsilon)$')

    texplot.savefig(fig,save_prefix+'Skew-Asymm-P')

def plot_shape_statistics_vs_press_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-vs-Press'

    # Remove 'mu' parameter
    kwargs.pop('P', None)

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'cnoidal', **kwargs})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=False)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = data_array

    fig = plot_shape_statistics_vs_depth_template(data_arrays,
            x_coordinate = 'P',
            ax_xlabel = r'Pressure $P k/(\rho_w g \epsilon)$')

    texplot.savefig(fig,save_prefix+'Skew-Asymm-Cnoidal-P')

def plot_energy(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'cnoidal', **kwargs},
            allow_multiple_files=True)

    data_array_list = []
    P_val_list = []
    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=False)

        P_val = float(data_array.attrs['P'])

        P_val_list.append(P_val)
        data_array_list.append(data_array)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = xr.concat(data_array_list, dim=xr.DataArray(P_val_list,
        name='P', dims='P'))
    # Transpose to put P coordinate at end; this makes plotting easier
    data_arrays[0] = data_arrays[0].transpose()
    # Remove P parameter attribute
    data_arrays[0].attrs.pop('P', None)

    fig = plot_energy_template(data_arrays)

    # Interpolate data
    for iy in range(data_arrays.shape[0]):
        data_arrays[iy] = data_arrays[iy].interpolate_na(
                't*eps*sqrt(g*h)*k_E', method='spline')

    # Plot best fit exponential
    ax = fig.axes[0]
    t = data_arrays[0]['t*eps*sqrt(g*h)*k_E']
    Ps = data_arrays[0]['P']
    # Get E
    energy = data_arrays[0]['E/E_0']
    # Fit with exponential
    fit = np.empty(Ps.size)
    variance = np.empty(Ps.size)
    for elem,_ in enumerate(Ps):
        if Ps[elem] == 0.0:
            # Can't calculate covariance for unforced case
            fit[elem] = np.nan
            variance[elem] = np.nan
            continue
        result = curve_fit(lambda t,b:
                np.exp(b*Ps[elem]*t), t, energy[:,elem], p0=(1/2))
        fit[elem] = result[0][0] # store slope
        variance[elem] = result[1][0,0] # store slope variance
    print('Mean exponential factor (Jeffreys, cnoidal): '+str(np.nanmean(fit)))
    print('Maximum STD of exponential factor: '+str(np.sqrt(np.nanmax(variance))))
    # Calculate energy using best fit
    energy_fit = np.exp(np.outer(t,Ps*fit))
    # Plot
    ax.plot(t, energy_fit, color='y', zorder=-1)

    texplot.savefig(fig,save_prefix+'Total-Energy-Jeffreys')

def plot_energy_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs},
            allow_multiple_files=True)

    data_array_list = []
    P_val_list = []
    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=False)

        P_val = float(data_array.attrs['P'])

        P_val_list.append(P_val)
        data_array_list.append(data_array)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = xr.concat(data_array_list, dim=xr.DataArray(P_val_list,
        name='P', dims='P'))
    # Transpose to put P coordinate at end; this makes plotting easier
    data_arrays[0] = data_arrays[0].transpose()
    # Remove P parameter attribute
    data_arrays[0].attrs.pop('P', None)

    fig = plot_energy_template(data_arrays)

    # Interpolate data
    for iy in range(data_arrays.shape[0]):
        data_arrays[iy] = data_arrays[iy].interpolate_na(
                't*eps*sqrt(g*h)*k_E', method='spline')

    # Plot best fit exponential
    ax = fig.axes[0]
    t = data_arrays[0]['t*eps*sqrt(g*h)*k_E']
    Ps = data_arrays[0]['P']
    # Get E
    energy = data_arrays[0]['E/E_0']
    # Fit with exponential
    fit = np.empty(Ps.size)
    variance = np.empty(Ps.size)
    for elem,_ in enumerate(Ps):
        if Ps[elem] == 0.0:
            # Can't calculate covariance for unforced case
            fit[elem] = np.nan
            variance[elem] = np.nan
            continue
        result = curve_fit(lambda t,b:
                1/(1-b*Ps[elem]*t)**2, t, energy[:,elem], p0=(1/5))
        fit[elem] = result[0][0] # store slope
        variance[elem] = result[1][0,0] # store slope variance
    print('Mean b factor (Jeffreys, solitary): '+str(np.nanmean(fit)))
    print('Maximum STD of exponential factor: '+str(np.sqrt(np.nanmax(variance))))
    # Calculate energy using best fit
    energy_fit = 1/(1-np.outer(t,Ps*fit))**2
    # Plot
    ax.plot(t, energy_fit, color='y', zorder=-1)

    texplot.savefig(fig,save_prefix+'Total-Energy-Jeffreys-Solitary')

def plot_energy_GM(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    kwargs['forcing_type'] = 'GM'

    filenames = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'cnoidal', **kwargs},
            allow_multiple_files=True)

    data_array_list = []
    P_val_list = []
    for filename in filenames:
        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=False)

        P_val = float(data_array.attrs['P'])

        P_val_list.append(P_val)
        data_array_list.append(data_array)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)
    data_arrays[0] = xr.concat(data_array_list, dim=xr.DataArray(P_val_list,
        name='P', dims='P'))
    # Transpose to put P coordinate at end; this makes plotting easier
    data_arrays[0] = data_arrays[0].transpose()
    # Remove P parameter attribute
    data_arrays[0].attrs.pop('P', None)

    fig = plot_energy_template(data_arrays)

    # Interpolate data
    for iy in range(data_arrays.shape[0]):
        data_arrays[iy] = data_arrays[iy].interpolate_na(
                't*eps*sqrt(g*h)*k_E', method='spline')

    # Plot best fit exponential
    ax = fig.axes[0]
    t = data_arrays[0]['t*eps*sqrt(g*h)*k_E']
    Ps = data_arrays[0]['P']
    # Get E
    energy = data_arrays[0]['E/E_0']
    # Fit with exponential
    from scipy.optimize import curve_fit
    fit = np.empty(Ps.size)
    variance = np.empty(Ps.size)
    for elem,_ in enumerate(Ps):
        if Ps[elem] == 0.0:
            # Can't calculate covariance for unforced case
            fit[elem] = np.nan
            variance[elem] = np.nan
            continue
        result = curve_fit(lambda t,b:
                np.exp(b*Ps[elem]*t), t, energy[:,elem], p0=(1/2))
        fit[elem] = result[0][0] # store slope
        variance[elem] = result[1][0,0] # store slope variance
    print('Mean exponential factor (GM, cnoidal): '+str(np.nanmean(fit)))
    print('Maximum STD of exponential factor: '+str(np.sqrt(np.nanmax(variance))))
    energy_fit = np.exp(np.outer(t,Ps*fit))
    # Plot
    ax.plot(t, energy_fit, color='y', zorder=-1)

    texplot.savefig(fig,save_prefix+'Total-Energy-GM')

def plot_power_spec_vs_kappa(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Power-Spec-vs-Kappa'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_power_spec_vs_kappa_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Power-Spectrum-Jeffreys')

def plot_power_spec_vs_kappa_GM(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Power-Spec-vs-Kappa'

    kwargs['forcing_type'] = 'GM'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val  in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_power_spec_vs_kappa_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Power-Spectrum-GM')

def plot_power_spec_vs_time(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Power-Spec-vs-Time'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_power_spec_vs_time_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Power-Spectrum-vs-Time-Jeffreys')

def plot_power_spec_vs_time_GM(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Power-Spec-vs-Time'

    kwargs['forcing_type'] = 'GM'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_power_spec_vs_time_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Power-Spectrum-vs-Time-GM')

def plot_wavenum_freq(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Wavenum-Freq'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_wavenum_freq_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Double-Power-Spectrum-Jeffreys')

def plot_wavenum_freq_GM(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Wavenum-Freq'

    kwargs['forcing_type'] = 'GM'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_wavenum_freq_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Double-Power-Spectrum-GM')

def plot_xt_offset_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'XT-Offset'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'solitary', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_xt_offset_template(data_arrays)

    texplot.savefig(fig,save_prefix+'XT-Offset')

def plot_xt_offset_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'XT-Offset'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    P = float(kwargs.get('P'))

    for indx_num, P_val in enumerate([P,-P]):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'P' : P_val})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_xt_offset_template(data_arrays)

    texplot.savefig(fig,save_prefix+'XT-Offset-Cnoidal')

def plot_biviscosity(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Biviscosity'

    # Remove 'nu_bi' parameter
    kwargs.pop('nu_bi', None)

    # Remove 'P' parameter
    kwargs.pop('P', None)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((1),dtype=object)

    mu = float(kwargs.get('mu'))

    for indx_num, mu_val in enumerate([mu]):
        filenames = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'mu' : mu_val},
                allow_multiple_files=True)

        data_array_list = []
        nu_bi_val_list = []
        for filename in filenames:
            # Extract data
            data_array = data_csv.load_data(filename, stack_coords=False)

            nu_bi_val = float(data_array.attrs['nu_bi'])

            nu_bi_val_list.append(nu_bi_val)
            data_array_list.append(data_array)

        data_arrays[indx_num] = xr.concat(data_array_list,
                dim=xr.DataArray(nu_bi_val_list, name='nu_bi', dims='nu_bi'))
        # Transpose to put P coordinate at end; this makes plotting easier
        data_arrays[indx_num] = data_arrays[indx_num].transpose()
        # Remove P parameter attribute
        data_arrays[indx_num].attrs.pop('nu_bi', None)

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            color_class='sequential', legend_sig_figs=3,
            legend_title=r'Biviscosity'+'\n'+\
                    r'$\nu/(\rho_w g k_E \epsilon)$',
            ax_title=r'$\epsilon = {eps}$, $\mu = {mu}$, '+\
                    r'$P k_E/(\rho_w g \epsilon)= 0$')

    texplot.savefig(fig,save_prefix+'Biviscosity')

def plot_decaying_no_nu_bi(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Decaying-no-NuBi'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((2),dtype=object)

    mu = float(kwargs.get('mu'))

    for indx_num, mu_val in enumerate(round_sig_figs([mu,7/8*mu])):
        filenames = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'mu' : mu_val},
                allow_multiple_files=True)

        data_array_list = []
        P_val_list = []
        for filename in filenames:
            # Extract data
            data_array = data_csv.load_data(filename, stack_coords=False)

            P_val = float(data_array.attrs['P'])

            P_val_list.append(P_val)
            data_array_list.append(data_array)

        data_arrays[indx_num] = xr.concat(data_array_list,
                dim=xr.DataArray(P_val_list, name='P', dims='P'))
        # Transpose to put P coordinate at end; this makes plotting easier
        data_arrays[indx_num] = data_arrays[indx_num].transpose()
        # Remove P parameter attribute
        data_arrays[indx_num].attrs.pop('P', None)

    fig = plot_shape_statistics_vs_time_template(data_arrays,
            color_class='inverse_sequential')

    texplot.savefig(fig,save_prefix+'Decaying-no-NuBi')

def plot_spacetime_mesh(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Full-Snapshots'

    # Using the default list of parameters, generate plots for ones with
    # slightly different parameters
    parameter_list = [
#            {**kwargs, 'label':''},
#            {**kwargs, 'forcing_type':'GM', 'label':'GM'},
            {**kwargs, 'wave_type':'solitary', 'label':'solitary'},
#            {**kwargs, 'P':-kwargs.get('P',0), 'label':'neg'},
#            {**kwargs, 'mu':round_sig_figs(7/8*kwargs.get('mu',0)), 'label':'double_mu'},
            ]
    for parameters in parameter_list:

        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters=parameters)

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        # Arrange data and parameters into 2d array for plotting
        data_arrays = np.empty((1,1),dtype=object)
        data_arrays[0,0] = data_array

        norm_by_wavelength = data_array.attrs['wave_type'] == 'cnoidal'

        fig = plot_spacetime_mesh_template(data_arrays,
                norm_by_wavelength=norm_by_wavelength)

        texplot.savefig(fig,save_prefix+'Spacetime-Mesh'+\
                ('_'+parameters['label'] if parameters['label'] != ''
                    else ''))

def plot_metrics(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Metrics'

    # Using the default list of parameters, generate plots for ones with
    # slightly different parameters
    parameter_list = [
            {**kwargs, 'wave_type':'solitary', 'label':'solitary'},
            ]
    for parameters in parameter_list:

        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters=parameters)

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=False,
                coord_names=['xStep','tStep','nu_bi'])

        data_array = data_array[{'nu_bi':0}]

        # Arrange data and parameters into 2d array for plotting
        data_arrays = np.empty((1,2),dtype=object)

        data_arrays[0,0] = data_array['Stopping time']
        data_arrays[0,1] = data_array['Normalized RMS change']

        data_arrays[0,0].attrs = data_array.attrs
        data_arrays[0,1].attrs = data_array.attrs

        norm_by_wavelength = data_array.attrs['wave_type'] == 'cnoidal'

        ax_title = np.array([['Stopping time',
            'Normalized RMS change']])

        fig = plot_metrics_mesh_template(data_arrays,
                norm_by_wavelength=norm_by_wavelength, suptitle='',
                ax_title=ax_title)

        texplot.savefig(fig,save_prefix+'Metrics'+\
                ('_'+parameters['label'] if parameters['label'] != ''
                    else ''))

def plot_forcing_types(load_prefix, save_prefix, *args, **kwargs):

    import scipy.special as spec
    from scipy.fftpack import diff as psdiff
    Height = 2
    P = 1

    WaveLength = 2*np.pi
    kwargs['WaveLength'] = WaveLength
    numWaves = 1
    xLen_forcing = WaveLength*numWaves
    dx_forcing = 0.01
    x = np.arange(0,xLen_forcing,dx_forcing)

    m = 0.1

    K = spec.ellipk(m)
    E = spec.ellipe(m)

    cn = spec.ellipj(x/WaveLength*2*K,m)[1]
    trough = Height/m*(1-m-E/K)
    y0 = trough + Height*cn**2

    # Convert eta' = eta/a = eta/h*eps to eta'/2 = eta/(2*a) = eta/H
    Profile = 1/2*y0
    Jeffreys = P*psdiff(Profile, period=xLen_forcing)
    GM = P*np.roll(Profile,
            shift=int(round(-float(kwargs.get('psi_P'))*WaveLength/(2*np.pi)/dx_forcing)),
            axis=0)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((3,1),dtype=object)

    for indx_num, plot in enumerate([Profile, Jeffreys, GM]):
        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = xr.DataArray(plot,
                dims=['x/lambda'],
                coords=[x])
        data_arrays[indx].attrs = kwargs

    fig = plot_forcing_types_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Forcing-Types')

def main():
    load_prefix = '../Data/Processed/'
    save_prefix = '../Reports/Figures/'

    # Remove default margins
    mpl.rcParams['axes.xmargin'] = 0

    default_params = {
            'eps' : 0.1,
            'mu' : 0.8,
            'P' : 0.25,
            'nu_bi' : 3e-3,
            'psi_P' : 3/4*np.pi,
            'forcing_type' : 'Jeffreys',
            }

    callable_functions = {
#            'trig_verf_no_nu_bi' : plot_trig_verf_no_nu_bi,
#            'trig_statistics_no_nu_bi' : plot_trig_statistics_no_nu_bi,
#            'long_verf_solitary_no_nu_bi' : plot_long_verf_solitary_no_nu_bi,
#            'long_verf_cnoidal_no_nu_bi' : plot_long_verf_cnoidal_no_nu_bi,
#            'long_statistics_solitary_no_nu_bi' : plot_long_statistics_solitary_no_nu_bi,
#            'long_statistics_cnoidal_no_nu_bi' : plot_long_statistics_cnoidal_no_nu_bi,
#            'trig_verf' : plot_trig_verf,
#            'trig_statistics' : plot_trig_statistics,
#            'long_verf_solitary' : plot_long_verf_solitary,
#            'long_verf_cnoidal' : plot_long_verf_cnoidal,
#            'long_statistics_solitary' : plot_long_statistics_solitary,
#            'long_statistics_cnoidal' : plot_long_statistics_cnoidal,
            'verf_solitary' : plot_verf_solitary,
            'pos_solitary' : plot_pos_solitary,
            'neg_solitary' : plot_neg_solitary,
            'pos_neg_solitary' : plot_pos_neg_solitary,
            'pos_neg_solitary_production' : plot_pos_neg_solitary_production,
            'pos_neg_solitary_terms' : plot_pos_neg_solitary_terms,
            'print_solitary_unforced_difference': print_solitary_unforced_difference,
            'pos_neg_solitary_tail' : plot_pos_neg_solitary_tail,
            'pos_neg_solitary_tail_thumbnail' : plot_pos_neg_solitary_tail_thumbnail,
            'pos_neg_solitary_and_sech' : plot_pos_neg_solitary_and_sech,
#            'pos_cnoidal' : plot_pos_cnoidal,
#            'neg_cnoidal' : plot_neg_cnoidal,
#            'pos_neg_cnoidal' : plot_pos_neg_cnoidal,
#            'pos_neg_cnoidal_GM' : plot_pos_neg_cnoidal_GM,
            'pos_neg_slope_solitary' : plot_pos_neg_slope_solitary,
#            'pos_neg_slope_cnoidal' : plot_pos_neg_slope_cnoidal,
#            'pos_neg_slope_cnoidal_GM' : plot_pos_neg_slope_cnoidal_GM,
            'slope_statistics_solitary' : plot_slope_statistics_solitary,
#            'slope_statistics_cnoidal' : plot_slope_statistics_cnoidal,
            'shape_statistics_solitary' : plot_shape_statistics_solitary,
            'shape_statistics_solitary_production' : plot_shape_statistics_solitary_production,
#            'shape_statistics_cnoidal' : plot_shape_statistics_cnoidal,
#            'shape_statistics_cnoidal_GM' : plot_shape_statistics_cnoidal_GM,
#            'shape_statistics_vs_depth' : plot_shape_statistics_vs_depth,
#            'shape_statistics_vs_press_solitary' : plot_shape_statistics_vs_press_solitary,
#            'shape_statistics_vs_press_cnoidal' : plot_shape_statistics_vs_press_cnoidal,
#            'energy' : plot_energy,
            'energy_solitary' : plot_energy_solitary,
#            'energy_GM' : plot_energy_GM,
#            'power_spec_vs_kappa' : plot_power_spec_vs_kappa,
#            'power_spec_vs_kappa_GM' : plot_power_spec_vs_kappa_GM,
#            'power_spec_vs_time' : plot_power_spec_vs_time,
#            'power_spec_vs_time_GM' : plot_power_spec_vs_time_GM,
#            'wavenum_freq' : plot_wavenum_freq,
#            'wavenum_freq_GM' : plot_wavenum_freq_GM,
            'xt_offset_solitary' : plot_xt_offset_solitary,
#            'xt_offset_cnoidal' : plot_xt_offset_cnoidal,
#            'biviscosity' : plot_biviscosity,
            'spacetime_mesh' : plot_spacetime_mesh,
#            'decaying_no_nu_bi' : plot_decaying_no_nu_bi,
            'metrics' : plot_metrics,
            'forcing_types' : plot_forcing_types,
            }

    if len(sys.argv) == 1:
        # No option provided; run all plots
        for function in callable_functions.values():
            function(load_prefix, save_prefix, **default_params)

    else:
        if not sys.argv[1] in callable_functions:
            raise ValueError('Command line option must be the name of '+
                'a callable function') from None
        if len(sys.argv) > 2:
            callable_functions[sys.argv[1]](load_prefix, save_prefix,
                    *sys.argv[2:])
        else:
            callable_functions[sys.argv[1]](load_prefix, save_prefix,
                    **default_params)

if __name__ == '__main__':
    main()
