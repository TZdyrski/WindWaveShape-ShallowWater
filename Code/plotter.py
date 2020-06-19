#!/usr/bin/env python3
# plotter.py
import sys
import glob
import numpy as np
import xarray as xr
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from pi_formatter import pi_multiple_ticks
import itertools
import texplot
import data_csv

def property_cycle(num_lines,color_class='sequential'):
    """Generates a property cycle.

    Parameters
    ----------
    num_lines : int
        Number of lines in the cycle.
    color_class : 'sequential' or 'cyclic'
        Type of color class to use for lines. The 'sequential' option
        uses the 'YlGnBu' color class. The 'cyclic' option uses the
        'twilight' color class. Default is 'sequential'.


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
            "color_class must be either 'sequential' or 'cyclic' but "+
            color_class+" was given"))

    prop_cycle = cycler('color', new_colors) + cycler('linestyle',
            linestyles)

    return prop_cycle

def annotate_arrow(ax, windLeft=True, wave_type='solitary'):
    if wave_type == 'solitary':
        arrowLeft = np.array([0.05,0.4])
        arrowRight = np.array([0.25,0.4])
        spacing = np.array([0.6,0])
    elif wave_type == 'cnoidal':
        arrowLeft = np.array([0.175,0.4])
        arrowRight = np.array([0.375,0.4])
        spacing = np.array([0.45,0])
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

def label_subplots(ax):

    if ax.size == 1:
        return
    for indx in np.ndindex(ax.shape):
        # Convert numerical index to lower case letter (0->a, 1->b, etc)
        indx_num = np.ravel_multi_index(indx,ax.shape)
        indxToAlpha = chr(indx_num+ord('a'))
        subplotLabel = indxToAlpha + ')'

        # Add subplot labels
        t = ax[indx].text(0.05, 0.9, subplotLabel,
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

def default_plotter(data_array, x_name, axis):
    axis.plot(data_array[x_name], data_array)

def plot_multiplot_template(data_arrays, x_coordinate, suptitle=None, ax_title=None,
        ax_xlabel=None, ax_ylabel=None, color_class=None,
        show_legend=False, legend_title=None, plotter=default_plotter):
    """
    Parameters
    ----------
    data_arrays : ndarray of xarrays
        An ndarray, dimension at most 2, with each element containing
        the xarray data_array for the corresponding subplot.
    """
    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=data_arrays.shape[0],
            ncols=data_arrays.shape[1], sharex=True,sharey='row')

    # Make 2d ndarrays even if only scalars or 1d arrays
    ax = atleast_2d(ax)

    # Adjust figure height
    figsize = fig.get_size_inches()
    fig.set_size_inches([figsize[0],figsize[1]*(0.7+0.3*ax.shape[0])])

    if suptitle is not None:
        fig.suptitle(suptitle)

    for iy, ix in np.ndindex(ax.shape):
        # Label axes and titles
        if ax_xlabel is not None:
            axes_sharing = ax[iy,ix].get_shared_x_axes().get_siblings(ax[iy,ix])
            if iy == ax.shape[0]-1 or len(axes_sharing) == 1:
                ax[iy,ix].set_xlabel(fill_to_shape(ax_xlabel,ax.shape)[iy,ix]\
                        .format(**data_arrays[iy,ix].attrs))
        if ax_ylabel is not None:
            axes_sharing = ax[iy,ix].get_shared_y_axes().get_siblings(ax[iy,ix])
            if ix == 0 or len(axes_sharing) == 1:
                ax[iy,ix].set_ylabel(fill_to_shape(ax_ylabel,ax.shape)[iy,ix]\
                        .format(**data_arrays[iy,ix].attrs))
        if ax_title is not None:
            ax[iy,ix].set_title(fill_to_shape(ax_title,ax.shape)[iy,ix]\
                    .format(**data_arrays[iy,ix].attrs))

        # Set property cycle
        prop_cycle = property_cycle(
                atleast_2d(data_arrays[iy,ix]).shape[1],
                **({'color_class':color_class} if color_class is not None
                    else {}),
                )
        ax[iy,ix].set_prop_cycle(prop_cycle)

        # Make 2d ndarrays even if only scalars or 1d arrays
        x_name = fill_to_shape(x_coordinate,ax.shape)[iy,ix]

        if atleast_2d(data_arrays[iy,ix]).shape[1] > 1:
            # Sort lines
            line_coord = [val for val in data_arrays[iy,ix].dims if val
                    != x_name][0]
            data_array_sorted = \
                    data_arrays[iy,ix].sortby(line_coord)
        else:
            # Only a single line, no need to sort
            data_array_sorted = data_arrays[iy,ix]
            line_coord = None

        # Plot snapshots
        plotter(data_array_sorted, x_name, ax[iy,ix])

    # Add subplot labels
    label_subplots(ax)

    if show_legend:
        # Add legend
        # Use values from first data_array since we assume they all use the same
        # values
        leg = fig.legend(np.around(data_arrays[0,0].sortby(line_coord)\
                [line_coord].values,1),
                **({'title':legend_title} if legend_title is not None else
                    {}),
                loc='right')
        # Set alignment in case provided legend_title is multilined
        leg.get_title().set_multialignment('center')

    subplot_adjust_params = {'hspace':0.3, 'wspace':0.2}
    if suptitle is not None:
        subplot_adjust_params['top'] = 0.825

    if show_legend is not None:
        subplot_adjust_params['right'] = 0.825

    if subplot_adjust_params != {}:
        fig.set_tight_layout(False)
        fig.tight_layout()
        fig.subplots_adjust(**subplot_adjust_params)

    # Make background transparent
    fig.patch.set_alpha(0)

    return fig

def plot_snapshots_template(data_arrays, norm_by_wavelength=True,
        wind_arrows=True, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Distance $x/\lambda$' if norm_by_wavelength else\
            r'Distance $x/h$'

    ax_ylabel = r'Wave Height $\eta / h$'

    suptitle = 'Wave Height vs Time' if data_arrays.size != 1 else None

    title_string =  r'$\epsilon = {eps}$, '+\
            r'$\mu = {mu}$'+'\n'+r'$P k_E/(\rho_w g \epsilon) = {P}$'

    if data_arrays.size == 1:
        title_string = 'Wave Height vs Time: ' + title_string

    ax_title = np.empty(data_arrays.shape,dtype=object)
    for iy, ix in np.ndindex(ax_title.shape):
        if iy == 0:
            ax_title[0,ix] = title_string
        else:
            ax_title[iy,ix] = r'$P k_E/(\rho_w g \epsilon) = {P}$'

    x_coordinate = 'x/lambda' if norm_by_wavelength else 'x/h'

    legend_title = r'Time'+'\n'+r'$t \epsilon \sqrt{g h} k_E$'

    if norm_by_wavelength:
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

    # Plot data
    fig = plot_multiplot_template(**{
        'data_arrays':data_arrays,
        'x_coordinate':x_coordinate,
        'suptitle':suptitle,
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        'show_legend':True,
        'legend_title':legend_title,
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    # Add arrow depicting wind and wave directions
    ax = atleast_2d(fig.axes).reshape(data_arrays.shape)
    if wind_arrows:
        for iy, ix in np.ndindex(ax.shape):
            annotate_arrow(ax[iy,ix], data_arrays[iy,ix].attrs['P']>=0,
                    wave_type=data_arrays[iy,ix].attrs['wave_type'])

    return fig

def plot_shape_statistics_template(data_arrays, ax_title=None, **kwargs):

    # Set axis labels and titles

    plot_biphase = 'biphase' in data_arrays[0].data_vars

    ax_ylabel = ['Height', 'Skewness', 'Asymmetry']
    if plot_biphase:
        ax_ylabel.append('Biphase')
    ax_ylabel = atleast_2d(np.array(ax_ylabel))

    if ax_title is not None:
        ax_title_full = fill_to_shape('', (len(ax_ylabel),data_arrays.size))
        ax_title_full[0,:] = ax_title

    if plot_biphase:
        data_arrays_rearranged = np.empty((4,data_arrays.size),dtype=object)
    else:
        data_arrays_rearranged = np.empty((3,data_arrays.size),dtype=object)

    for shape_group in range(data_arrays_rearranged.shape[1]):
        split_data_arrays = [
                data_arrays[shape_group]['max(eta)/max(eta_0)'],
                data_arrays[shape_group]['skewness'],
                data_arrays[shape_group]['asymmetry'],
                ]
        if plot_biphase:
            split_data_arrays.append(data_arrays[shape_group]['biphase'])

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

    ax = atleast_2d(fig.axes).reshape((len(ax_ylabel),data_arrays.size))
    for ix in np.ndindex(ax.shape[1]):
        # Put horizontal line at y=1
        ax[0,ix].item().axhline(1, color='0.75')

        # Put horizontal line at y=0
        ax[2,ix].item().axhline(0, color='0.75')

        if plot_biphase:
            # Put horizontal line at y=0
            ax[3,ix].item().axhline(0, color='0.75')

            # Determine smallest ylim (positive or negative)
            min_ylim = np.amin(np.abs(np.array(\
                    ax[3,ix].item().get_ylim())))
            # Don't let min_ylim be smaller than 1/2 max_ylim
            minSize = 1/2*np.amax(np.abs(np.array(\
                    ax[3,ix].item().get_ylim())))
            min_ylim = max([minSize,min_ylim])
            # Determine smallest power of 2, n, such that min_ylim >=
            # pi/2^n
            power_of_two = np.ceil(np.log(np.pi/min_ylim)/np.log(2))
            # Denominator of pi is 2^(power_of_two)
            pi_denom = 2**power_of_two
            # Set y-ticks as multiples of \pi
            pi_multiple_ticks(ax[3,ix].item(),'y',1/pi_denom,1/(2*pi_denom))

    return fig

def plot_shape_statistics_vs_time_template(data_arrays, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Time $t \epsilon \sqrt{{g h}} k_E$'

    suptitle = 'Shape Statistics vs Time' if data_arrays.size != 1 else None

    title_string =  r'$\epsilon = {eps}$, $\mu = {mu}$'

    if data_arrays.size == 1:
        title_string = 'Shape Statistics vs Time: ' + title_string

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
                'suptitle':suptitle,
                'show_legend':True,
                'legend_title':legend_title,
                **kwargs})

    return fig

def plot_shape_statistics_vs_depth_template(data_arrays, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Depth $k_E h$'

    suptitle = 'Shape Statistics vs Time' if data_arrays.size != 1 else None

    title_string =  r'$\epsilon = {eps}$, '+\
            r'$t \epsilon \sqrt{{g h}} k_E = {t*eps*sqrt(g*h)*k_E}$'

    if data_arrays.size == 1:
        title_string = 'Shape Statistics vs Time: ' + title_string

    ax_title = np.empty(data_arrays.shape,dtype=object)
    for ix in np.ndindex(ax_title.shape):
        ax_title[ix] = title_string

    x_coordinate = 'k_E*h'

    fig = plot_shape_statistics_template(data_arrays,
            **{
                'ax_xlabel':ax_xlabel,
                'ax_title':ax_title,
                'x_coordinate':x_coordinate,
                'suptitle':suptitle,
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

    # Use parameters from first data_array since we assume they're all
    # the same
    title_string = r'Total Energy: $\epsilon = {eps}$, $\mu = {mu}$'.format(
            **data_arrays[0].attrs)
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
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        'show_legend':True,
        'legend_title':legend_title,
        'color_class':'cyclic',
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    ax = atleast_2d(fig.axes)
    for ix in np.ndindex(ax.shape[1]):
        # Put horizontal line at y=1
        ax[0,ix].item().axhline(1, color='0.75')

    return fig

def plot_power_spec_vs_kappa_template(data_arrays, **kwargs):

    # Set axis labels and titles

    ax_xlabel = r'Harmonic $\kappa/k$'

    # Plot power spectrum \abs{\hat'{eps*eta'}}^2 = \abs{\hat{eta}}^2*k^2_E/h^2
    # (Primes denote the nondim variables used throughout this solver)
    ax_ylabel = r'Energy $\abs{{\hat{{\eta}}}}^2 k_E^2/h^2$'

    # Use parameters from first data_array since we assume they're all
    # the same
    suptitle = r'Power Spectrum vs Kappa: $\epsilon={eps}$, $\mu = {mu}$'.\
            format(**data_arrays[0,0].attrs)

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

    # Set axis labels and titles

    ax_xlabel = r'Time $t \epsilon \sqrt{{g h}} k_E$'

    ax_ylabel = r'\begin{{tabular}}{{c}} Normalized \\'+\
            r'Energy $\abs{{\hat{{\eta}}}}^2 k_E^2/h^2$\end{{tabular}}'

    # Use parameters from first data_array since we assume they're all
    # the same
    suptitle = r'Power Spectrum vs Time: $\epsilon = {eps}$, $\mu = {mu}$'.format(
            **data_arrays[0,0].attrs)

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

    ax_ylabel = r'Frequency $\omega/\sqrt{{g h}} k_E$'

    # Use parameters from first data_array since we assume they're all
    # the same
    suptitle = r'Wavenumber Frequency Plot of $\abs{{\hat{{\eta}}}}^2'+\
            r' k_E^4 g / h$: $\epsilon = {eps}$, $\mu = {mu}$'.format(
            **data_arrays[0,0].attrs)

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
                data_array['omega/sqrt(g*h)/k'],
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
        'ax_title':ax_title,
        'ax_xlabel':ax_xlabel,
        'ax_ylabel':ax_ylabel,
        'plotter':contour_plotter,
        # Put kwargs last so any parameters will overwrite the defaults
        # we've provided
        **kwargs,
        })

    return fig

def plot_forcing_types_template(data_arrays):
    # Initialize figure
    fig, ax = texplot.newfig(0.9,nrows=2,ncols=1,sharex=True,
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

def plot_trig_verf(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'TrigVerf'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    # Extract data
    for indx_num, solver in enumerate(['Builtin', 'RK3']):
        filename = data_csv.find_filenames(load_prefix, filename_base,
                required_words=[solver])

        data = data_csv.load_data(filename, stack_coords=True)

        # Add kh = sqrt(mu) to attrs
        data.attrs['kh'] = round(np.sqrt(data.attrs['mu']),1)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data

    # Make 2d ndarrays even if only scalars or 1d arrays
    data_arrays = np.atleast_2d(np.array(data_arrays))

    title_string = r'{solver} Solver after exactly 1 Period: '+\
                '$a_0/h={eps}$, $k_E h = {kh}$'

    fig = plot_snapshots_template(data_arrays, suptitle=None,
            ax_ylabel=np.array([[''],['']]),
            ax_title=np.array([[title_string],[title_string]]),
            wind_arrows=False)

    texplot.savefig(fig,save_prefix+filename_base)

def plot_long_verf_solitary(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'LongVerf'

    filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type':'solitary'})

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
            parameters={'wave_type':'cnoidal'})

    # Extract data
    data_array = data_csv.load_data(filename, stack_coords=True)

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((1,1),dtype=object)
    data_arrays[0,0] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Long-Run-Cnoidal')

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

    for indx_num, Psign in enumerate([1,-1]):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
            parameters={'wave_type' : 'solitary', **kwargs})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_snapshots_template(data_arrays, norm_by_wavelength=False)

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative')

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

    for indx_num, (Psign, mu_val) in enumerate(itertools.product([1,-1],[mu,2*mu])):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs,
                    'mu' : mu_val})

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

    for indx_num, Psign in enumerate([1,-1]):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_snapshots_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Snapshots-Positive-Negative-Cnoidal-GM')

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

def plot_shape_statistics_cnoidal(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Shape-Statistics'

    # Remove 'P' parameter
    kwargs.pop('P', None)

    # Arrange data and parameters into 1d array for plotting
    data_arrays = np.empty((2),dtype=object)

    mu = float(kwargs.get('mu'))

    for indx_num, mu_val in enumerate([mu,2*mu]):
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

    texplot.savefig(fig,save_prefix+'Skew-Asymm-Cnoidal')

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

    texplot.savefig(fig,save_prefix+'Total-Energy-Jeffreys')

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

    texplot.savefig(fig,save_prefix+'Total-Energy-GM')

def plot_power_spec_vs_kappa(load_prefix, save_prefix, *args, **kwargs):
    filename_base = 'Power-Spec-vs-Kappa'

    # Arrange data and parameters into 2d array for plotting
    data_arrays = np.empty((2,1),dtype=object)

    for indx_num, Psign in enumerate([1,-1]):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs})

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

    for indx_num, Psign in enumerate([1,-1]):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs})

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

    for indx_num, Psign in enumerate([1,-1]):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs})

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

    for indx_num, Psign in enumerate([1,-1]):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs})

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

    for indx_num, Psign in enumerate([1,-1]):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs})

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

    for indx_num, Psign in enumerate([1,-1]):
        kwargs['P'] = Psign*kwargs['P']
        filename = data_csv.find_filenames(load_prefix, filename_base,
                parameters={'wave_type' : 'cnoidal', **kwargs})

        # Extract data
        data_array = data_csv.load_data(filename, stack_coords=True)

        indx = np.unravel_index(indx_num,data_arrays.shape)
        data_arrays[indx] = data_array

    fig = plot_wavenum_freq_template(data_arrays)

    texplot.savefig(fig,save_prefix+'Double-Power-Spectrum-GM')

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

    default_params = {
            'eps' : 0.1,
            'mu' : 0.8,
            'P' : 0.25,
            'H' : 0.25*0.05,
            'psi_P' : 3/4*np.pi,
            'forcing_type' : 'Jeffreys',
            }

    callable_functions = {
            'trig_verf' : plot_trig_verf,
            'long_verf_solitary' : plot_long_verf_solitary,
            'long_verf_cnoidal' : plot_long_verf_cnoidal,
            'pos_solitary' : plot_pos_solitary,
            'neg_solitary' : plot_neg_solitary,
            'pos_neg_solitary' : plot_pos_neg_solitary,
            'pos_cnoidal' : plot_pos_cnoidal,
            'neg_cnoidal' : plot_neg_cnoidal,
            'pos_neg_cnoidal' : plot_pos_neg_cnoidal,
            'pos_neg_cnoidal_GM' : plot_pos_neg_cnoidal_GM,
            'shape_statistics_solitary' : plot_shape_statistics_solitary,
            'shape_statistics_cnoidal' : plot_shape_statistics_cnoidal,
            'shape_statistics_vs_depth' : plot_shape_statistics_vs_depth,
            'energy' : plot_energy,
            'energy_GM' : plot_energy_GM,
            'power_spec_vs_kappa' : plot_power_spec_vs_kappa,
            'power_spec_vs_kappa_GM' : plot_power_spec_vs_kappa_GM,
            'power_spec_vs_time' : plot_power_spec_vs_time,
            'power_spec_vs_time_GM' : plot_power_spec_vs_time_GM,
            'wavenum_freq' : plot_wavenum_freq,
            'wavenum_freq_GM' : plot_wavenum_freq_GM,
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
